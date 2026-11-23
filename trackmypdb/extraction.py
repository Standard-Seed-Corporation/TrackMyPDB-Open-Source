"""
Heteroatom (ligand) extraction from the PDB structures mapped to UniProt
proteins.

Workflow for ``HeteroatomExtractor.extract``:

1. UniProt accession -> PDB entries, using the PDBe "best structures" mapping
   (kept in PDBe's quality order, best first).
2. Every structure is downloaded (mmCIF, falling back to legacy PDB format) in
   parallel and its ``HETATM`` records are grouped by chemical-component code.
3. Each *unique* component code is resolved once to SMILES/name/formula via the
   RCSB chemical-component API, then the RCSB ideal-coordinate SDF, then the
   bundled offline ligand database.
4. Water, ions, buffers, cryo-protectants and polymer-linking residues
   (e.g. selenomethionine) are removed, and one record per
   (UniProt, PDB, ligand) is returned as a DataFrame.

No Streamlit code lives here: progress is reported through an optional
``progress(fraction, message)`` callback and problems through ``logging``.
"""

from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Iterable

import pandas as pd
import requests

from .http import DEFAULT_TIMEOUT, get_session
from .ligand_db import ligand_lookup

log = logging.getLogger("trackmypdb")

PDBE_BEST_STRUCTURES_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{uniprot}"
RCSB_MMCIF_URL = "https://files.rcsb.org/download/{pdb}.cif"
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb}.pdb"
RCSB_CHEMCOMP_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{code}"
RCSB_LIGAND_SDF_URL = "https://files.rcsb.org/ligands/download/{code}_ideal.sdf"

ProgressCallback = Callable[[float, str], None]

#: Components that are almost never the ligand of interest: water, common
#: ions, crystallisation buffers, cryo-protectants and detergents.
EXCLUDED_CODES: frozenset[str] = frozenset({
    # water
    "HOH", "WAT", "H2O", "DOD", "D2O", "DIS",
    # ions
    "NA", "K", "LI", "RB", "CS", "MG", "CA", "SR", "BA", "ZN", "FE", "FE2", "CU", "CU1",
    "MN", "CO", "NI", "CD", "HG", "PT", "AU", "AG", "PB", "AL", "GA",
    "CL", "BR", "IOD", "I", "F", "SO4", "SUL", "PO4", "PI", "NO3", "NH4", "SCN",
    "AZI", "CO3", "BCT", "OH", "UNX", "UNL",
    # buffers, solvents, cryo-protectants, detergents
    "ACT", "ACY", "ACE", "FMT", "EDO", "GOL", "PEG", "PGE", "PG4", "P6G", "PE4", "PE3",
    "PE8", "1PE", "2PE", "12P", "15P", "MPD", "DMS", "DMF", "BME", "TRS", "MES", "EPE",
    "HEP", "IMD", "CIT", "FLC", "TAR", "TLA", "MLI", "MLA", "SIN", "BU3", "IPA", "EOH",
    "MOH", "ACN", "BOG", "LDA", "SDS", "C8E", "DTT", "DTU", "TCE", "PEO",
})

#: Component types (from the Chemical Component Dictionary) that are parts of a
#: polymer chain rather than bound ligands, e.g. modified amino acids.
POLYMER_COMPONENT_TYPES = ("PEPTIDE LINKING", "PEPTIDE-LIKE", "RNA LINKING", "DNA LINKING",
                           "L-GAMMA-PEPTIDE", "L-BETA-PEPTIDE")

RECORD_COLUMNS = [
    "UniProt_ID", "PDB_ID", "Heteroatom_Code", "Chemical_Name", "SMILES", "Formula",
    "Component_Type", "InChIKey", "Source", "Chains", "Residue_Numbers", "Copies",
    "Atom_Count",
]

_UNIPROT_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d+)?$")
_CIF_TOKEN_RE = re.compile(r"'(?:[^']|'(?=\S))*'|\"[^\"]*\"|\S+")


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #

def is_valid_uniprot_accession(accession: str) -> bool:
    """True if ``accession`` matches the UniProt accession format
    (isoform suffixes such as ``P04637-2`` are accepted)."""
    return bool(_UNIPROT_RE.match((accession or "").strip().upper()))


def parse_uniprot_ids(text: str | Iterable[str]) -> list[str]:
    """Split comma/space/newline separated input into unique, upper-cased IDs,
    preserving order."""
    if isinstance(text, str):
        tokens = re.split(r"[\s,;]+", text)
    else:
        tokens = list(text)
    seen: dict[str, None] = {}
    for tok in tokens:
        tok = (tok or "").strip().upper()
        if tok:
            seen.setdefault(tok, None)
    return list(seen)


# --------------------------------------------------------------------------- #
# Structure parsing
# --------------------------------------------------------------------------- #

@dataclass
class HetGroup:
    """All HETATM records of one component code within one structure."""
    chains: set = field(default_factory=set)
    residues: set = field(default_factory=set)      # (chain, residue number)
    atom_names: set = field(default_factory=set)

    def add(self, chain: str, resnum: str, atom: str) -> None:
        if chain:
            self.chains.add(chain)
        self.residues.add((chain, resnum))
        if atom:
            self.atom_names.add(atom)


def parse_pdb_hetatm(lines: Iterable[str]) -> dict[str, HetGroup]:
    """Group legacy-format ``HETATM`` records by residue name (columns 18-20)."""
    groups: dict[str, HetGroup] = {}
    for line in lines:
        if not line.startswith("HETATM"):
            continue
        code = line[17:20].strip()
        if not code:
            continue
        groups.setdefault(code, HetGroup()).add(
            chain=line[21:22].strip(), resnum=line[22:27].strip(), atom=line[12:16].strip())
    return groups


def parse_mmcif_hetatm(text: str) -> dict[str, HetGroup]:
    """Group ``HETATM`` rows of the mmCIF ``_atom_site`` loop by component code.

    mmCIF is the only format available for very large entries (most cryo-EM
    structures) and for components with 5-character codes.
    """
    groups: dict[str, HetGroup] = {}
    fields: list[str] = []
    in_header = False
    in_atom_site = False
    for raw in text.splitlines():
        line = raw.strip()
        if line == "loop_":
            in_header, in_atom_site, fields = True, False, []
            continue
        if in_header and line.startswith("_atom_site."):
            fields.append(line.split(".", 1)[1].split()[0])
            continue
        if in_header and fields:
            in_header, in_atom_site = False, True  # first data row
        elif in_header:
            in_header = False
        if not in_atom_site:
            continue
        if not line or line.startswith(("#", "_", "loop_", "data_")):
            in_atom_site = False
            continue
        if not line.startswith("HETATM"):
            continue
        tokens = _CIF_TOKEN_RE.findall(line)
        if len(tokens) < len(fields):
            continue
        row = dict(zip(fields, tokens))

        def val(*names):
            for n in names:
                v = row.get(n)
                if v not in (None, "?", "."):
                    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                        v = v[1:-1]
                    return v
            return ""

        code = val("auth_comp_id", "label_comp_id")
        if not code:
            continue
        groups.setdefault(code, HetGroup()).add(
            chain=val("auth_asym_id", "label_asym_id"),
            resnum=val("auth_seq_id", "label_seq_id"),
            atom=val("auth_atom_id", "label_atom_id"),
        )
    return groups


# --------------------------------------------------------------------------- #
# Ligand information
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LigandInfo:
    smiles: str = ""
    name: str = ""
    formula: str = ""
    component_type: str = ""
    inchikey: str = ""
    source: str = "not_found"

    @property
    def is_polymer_component(self) -> bool:
        t = self.component_type.upper()
        return any(p in t for p in POLYMER_COMPONENT_TYPES)


@dataclass
class ExtractionResult:
    """Output of :meth:`HeteroatomExtractor.extract`."""
    records: pd.DataFrame
    pdbs_per_uniprot: dict[str, list[str]] = field(default_factory=dict)
    failed_structures: list[str] = field(default_factory=list)
    invalid_ids: list[str] = field(default_factory=list)
    unmapped_ids: list[str] = field(default_factory=list)

    @property
    def summary(self) -> dict:
        df = self.records
        return {
            "uniprot_ids": list(self.pdbs_per_uniprot),
            "structures_requested": sum(len(v) for v in self.pdbs_per_uniprot.values()),
            "structures_failed": len(self.failed_structures),
            "records": int(len(df)),
            "records_with_smiles": int((df["SMILES"] != "").sum()) if len(df) else 0,
            "unique_ligands": int(df["Heteroatom_Code"].nunique()) if len(df) else 0,
            "invalid_ids": self.invalid_ids,
            "ids_without_structures": self.unmapped_ids,
        }


class HeteroatomExtractor:
    """Extract ligands from the PDB structures of UniProt proteins.

    Parameters
    ----------
    workers:
        Number of parallel HTTP workers (keep it modest to respect the public APIs).
    use_local_db:
        Fall back to the bundled ligand database when online lookups fail.
    exclude_codes:
        Component codes to drop (water, ions, buffers...). Pass ``frozenset()``
        to keep everything.
    exclude_polymer_components:
        Drop components that are part of a polymer chain (modified residues).
    session_factory:
        Callable returning a ``requests.Session``-like object; mainly for tests.
    """

    def __init__(self, workers: int = 6, use_local_db: bool = True,
                 exclude_codes: Iterable[str] = EXCLUDED_CODES,
                 exclude_polymer_components: bool = True,
                 timeout: float = DEFAULT_TIMEOUT,
                 session_factory: Callable[[], requests.Session] | None = None):
        self.workers = max(1, int(workers))
        self.use_local_db = use_local_db
        self.exclude_codes = frozenset(c.upper() for c in exclude_codes)
        self.exclude_polymer_components = exclude_polymer_components
        self.timeout = timeout
        self._session = session_factory or get_session
        self._ligand_cache: dict[str, LigandInfo] = {}
        self._cache_lock = threading.Lock()

    # ---------------------------------------------------------------- HTTP --
    def _get(self, url: str) -> requests.Response:
        return self._session().get(url, timeout=self.timeout)

    def get_pdbs_for_uniprot(self, uniprot: str) -> list[str]:
        """PDB IDs mapped to a UniProt accession, best structures first.

        Returns an empty list if the accession has no mapped structures.
        Raises ``requests.RequestException`` on network/server errors.
        """
        uniprot = uniprot.strip().upper()
        r = self._get(PDBE_BEST_STRUCTURES_URL.format(uniprot=uniprot))
        if r.status_code == 404:
            return []
        r.raise_for_status()
        data = r.json()
        structs = []
        if isinstance(data, dict):
            val = data.get(uniprot) or next(iter(data.values()), [])
            structs = val.get("best_structures", []) if isinstance(val, dict) else val
        elif isinstance(data, list):
            structs = data
        ordered: dict[str, None] = {}
        for s in structs or []:
            pdb = (s.get("pdb_id") or "").upper() if isinstance(s, dict) else ""
            if pdb:
                ordered.setdefault(pdb, None)
        return list(ordered)

    def fetch_structure_hets(self, pdb_id: str) -> dict[str, HetGroup]:
        """Download a structure (mmCIF first, then legacy PDB) and return its
        HETATM groups. Raises ``requests.RequestException`` if both fail."""
        pdb_id = pdb_id.upper()
        try:
            r = self._get(RCSB_MMCIF_URL.format(pdb=pdb_id))
            r.raise_for_status()
            return parse_mmcif_hetatm(r.text)
        except requests.RequestException as exc:
            log.info("mmCIF download failed for %s (%s); trying PDB format", pdb_id, exc)
        r = self._get(RCSB_PDB_URL.format(pdb=pdb_id))
        r.raise_for_status()
        return parse_pdb_hetatm(r.text.splitlines())

    # ------------------------------------------------------------- ligands --
    def _from_rcsb(self, code: str) -> LigandInfo | None:
        r = self._get(RCSB_CHEMCOMP_URL.format(code=code))
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        desc = data.get("rcsb_chem_comp_descriptor") or {}
        comp = data.get("chem_comp") or {}
        return LigandInfo(
            smiles=desc.get("smiles_stereo") or desc.get("smiles") or "",
            name=comp.get("name") or "",
            formula=comp.get("formula") or "",
            component_type=comp.get("type") or "",
            inchikey=desc.get("in_ch_ikey") or desc.get("inchi_key") or "",
            source="rcsb",
        )

    def _from_rcsb_sdf(self, code: str) -> str:
        try:
            from rdkit import Chem
        except ImportError:
            return ""
        r = self._get(RCSB_LIGAND_SDF_URL.format(code=code))
        if r.status_code != 200:
            return ""
        mol = Chem.MolFromMolBlock(r.text)
        return Chem.MolToSmiles(mol) if mol is not None else ""

    def lookup_ligand(self, code: str) -> LigandInfo:
        """Resolve a chemical-component code to SMILES and metadata (cached)."""
        code = code.strip().upper()
        with self._cache_lock:
            if code in self._ligand_cache:
                return self._ligand_cache[code]

        info: LigandInfo | None = None
        try:
            info = self._from_rcsb(code)
            if info is not None and not info.smiles:
                smiles = self._from_rcsb_sdf(code)
                if smiles:
                    info = LigandInfo(smiles, info.name, info.formula, info.component_type,
                                      info.inchikey, "rcsb_sdf")
        except (requests.RequestException, ValueError) as exc:
            log.warning("Online lookup failed for %s: %s", code, exc)

        if (info is None or not info.smiles) and self.use_local_db:
            hit = ligand_lookup().get(code)
            if hit:
                info = LigandInfo(
                    smiles=hit["SMILES"],
                    name=(info.name if info and info.name else hit["Chemical_Name"]),
                    formula=(info.formula if info and info.formula else hit["Formula"]),
                    component_type=(info.component_type if info and info.component_type
                                    else hit["Component_Type"]),
                    inchikey=info.inchikey if info else "",
                    source="local_db",
                )
        info = info or LigandInfo()
        with self._cache_lock:
            self._ligand_cache[code] = info
        return info

    # ------------------------------------------------------------ pipeline --
    def extract(self, uniprot_ids: str | Iterable[str], max_structures: int | None = None,
                progress: ProgressCallback | None = None) -> ExtractionResult:
        """Run the full extraction for one or more UniProt accessions.

        ``max_structures`` caps the number of PDB entries per accession
        (best-ranked first); ``None`` or ``0`` means no cap.
        """
        ids = parse_uniprot_ids(uniprot_ids)
        invalid = [u for u in ids if not is_valid_uniprot_accession(u)]
        ids = [u for u in ids if u not in invalid]
        result = ExtractionResult(records=pd.DataFrame(columns=RECORD_COLUMNS),
                                  invalid_ids=invalid)

        def report(frac: float, msg: str) -> None:
            if progress:
                progress(min(max(frac, 0.0), 1.0), msg)

        # 1. UniProt -> PDB
        for i, up in enumerate(ids):
            report(0.05 * i / max(len(ids), 1), f"Finding structures for {up}")
            try:
                pdbs = self.get_pdbs_for_uniprot(up)
            except (requests.RequestException, ValueError) as exc:
                log.warning("PDBe mapping failed for %s: %s", up, exc)
                pdbs = []
            if max_structures:
                pdbs = pdbs[:max_structures]
            if not pdbs:
                result.unmapped_ids.append(up)
            result.pdbs_per_uniprot[up] = pdbs

        unique_pdbs = list(dict.fromkeys(p for v in result.pdbs_per_uniprot.values() for p in v))
        if not unique_pdbs:
            report(1.0, "No PDB structures found")
            return result

        # 2. Download + parse structures in parallel
        hets: dict[str, dict[str, HetGroup]] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self.fetch_structure_hets, p): p for p in unique_pdbs}
            for n, fut in enumerate(as_completed(futures), 1):
                pdb = futures[fut]
                try:
                    hets[pdb] = fut.result()
                except Exception as exc:  # noqa: BLE001 - keep going on any failure
                    log.warning("Could not download/parse %s: %s", pdb, exc)
                    result.failed_structures.append(pdb)
                report(0.05 + 0.6 * n / len(unique_pdbs),
                       f"Downloaded {n}/{len(unique_pdbs)} structures")

        # 3. Resolve each unique component code once
        codes = sorted({c for groups in hets.values() for c in groups
                        if c.upper() not in self.exclude_codes})
        infos: dict[str, LigandInfo] = {}
        if codes:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(self.lookup_ligand, c): c for c in codes}
                for n, fut in enumerate(as_completed(futures), 1):
                    infos[futures[fut]] = fut.result()
                    report(0.65 + 0.35 * n / len(codes), f"Resolved {n}/{len(codes)} ligands")

        # 4. Build records
        rows = []
        for up, pdbs in result.pdbs_per_uniprot.items():
            for pdb in pdbs:
                for code, grp in sorted(hets.get(pdb, {}).items()):
                    info = infos.get(code)
                    if info is None:
                        continue  # excluded code
                    if self.exclude_polymer_components and info.is_polymer_component:
                        continue
                    rows.append({
                        "UniProt_ID": up,
                        "PDB_ID": pdb,
                        "Heteroatom_Code": code,
                        "Chemical_Name": info.name,
                        "SMILES": info.smiles,
                        "Formula": info.formula,
                        "Component_Type": info.component_type,
                        "InChIKey": info.inchikey,
                        "Source": info.source,
                        "Chains": ", ".join(sorted(grp.chains)),
                        "Residue_Numbers": ", ".join(
                            sorted({r for _, r in grp.residues if r}, key=_natural_key)),
                        "Copies": len(grp.residues),
                        "Atom_Count": len(grp.atom_names),
                    })
        if rows:
            result.records = pd.DataFrame(rows, columns=RECORD_COLUMNS)
        report(1.0, "Extraction complete")
        return result


def _natural_key(s: str):
    m = re.match(r"(-?\d+)(.*)", s)
    return (int(m.group(1)), m.group(2)) if m else (float("inf"), s)
