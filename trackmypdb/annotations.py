"""
Protein annotations for result tables:

* PDB entry -> UniProt accessions and protein names (RCSB GraphQL, batched).
* UniProt accession -> disease involvement (UniProtKB ``DISEASE`` comments).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable

import pandas as pd
import requests

from .http import DEFAULT_TIMEOUT, get_session

log = logging.getLogger("trackmypdb")

RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"
UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"

_ENTRIES_QUERY = """
query ($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    polymer_entities {
      rcsb_polymer_entity { pdbx_description }
      rcsb_polymer_entity_container_identifiers {
        uniprot_ids
        reference_sequence_identifiers { database_accession database_name }
      }
    }
  }
}
"""

#: Keyword groups used to filter disease annotations.
DISEASE_CATEGORIES: dict[str, list[str]] = {
    "Cancer": ["cancer", "carcinoma", "tumor", "tumour", "oncogene", "malignan", "neoplasm",
               "leukemia", "leukaemia", "lymphoma", "sarcoma", "melanoma", "glioma"],
    "Diabetes": ["diabetes", "diabetic", "insulin resistance", "hyperglycemia", "mody"],
    "Cardiovascular": ["cardiovascular", "heart", "cardiac", "cardiomyopathy", "hypertension",
                       "atherosclerosis", "coronary", "arrhythmia"],
    "Neurological": ["alzheimer", "parkinson", "dementia", "neurodegenera", "epilep",
                     "multiple sclerosis", "ataxia", "neuropathy"],
    "Immune": ["immune", "autoimmune", "inflammat", "arthritis", "lupus", "immunodeficiency"],
    "Infectious": ["viral", "bacterial", "infection", "hiv", "tuberculosis", "hepatitis", "covid"],
    "Metabolic": ["metabolic", "obesity", "lipid", "cholesterol", "fatty acid", "dyslipidemia",
                  "storage disease"],
}

ProgressCallback = Callable[[float, str], None]


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def fetch_pdb_protein_info(pdb_ids: Iterable[str], batch_size: int = 100,
                           session: requests.Session | None = None,
                           progress: ProgressCallback | None = None) -> dict[str, dict]:
    """Map PDB IDs to ``{"uniprot_ids": [...], "protein_names": [...]}`` using
    batched RCSB GraphQL queries (one request per ``batch_size`` entries).
    IDs that could not be fetched map to empty lists."""
    ids = list(dict.fromkeys(p.upper() for p in pdb_ids if p and p != "N/A"))
    session = session or get_session()
    info = {p: {"uniprot_ids": [], "protein_names": []} for p in ids}
    batches = list(_chunks(ids, batch_size))
    for n, batch in enumerate(batches, 1):
        try:
            r = session.post(RCSB_GRAPHQL_URL, json={"query": _ENTRIES_QUERY,
                                                     "variables": {"ids": batch}},
                             timeout=DEFAULT_TIMEOUT)
            r.raise_for_status()
            entries = (r.json().get("data") or {}).get("entries") or []
        except (requests.RequestException, ValueError) as exc:
            log.warning("RCSB GraphQL batch failed: %s", exc)
            entries = []
        for entry in entries:
            if not entry:
                continue
            rec = info.setdefault(entry["rcsb_id"].upper(), {"uniprot_ids": [], "protein_names": []})
            for ent in entry.get("polymer_entities") or []:
                ids_block = ent.get("rcsb_polymer_entity_container_identifiers") or {}
                accs = list(ids_block.get("uniprot_ids") or [])
                accs += [ref.get("database_accession") for ref in
                         ids_block.get("reference_sequence_identifiers") or []
                         if ref.get("database_name") == "UniProt"]
                for acc in accs:
                    if acc and acc not in rec["uniprot_ids"]:
                        rec["uniprot_ids"].append(acc)
                desc = (ent.get("rcsb_polymer_entity") or {}).get("pdbx_description")
                if desc and desc not in rec["protein_names"]:
                    rec["protein_names"].append(desc)
        if progress:
            progress(n / len(batches), f"Fetched protein info batch {n}/{len(batches)}")
    return info


def add_protein_info(df: pd.DataFrame, pdb_column: str = "PDB_ID",
                     progress: ProgressCallback | None = None) -> pd.DataFrame:
    """Return a copy of ``df`` with ``UniProt_IDs`` and ``Protein_Names`` columns."""
    out = df.copy()
    info = fetch_pdb_protein_info(out[pdb_column].astype(str).tolist(), progress=progress)
    out["UniProt_IDs"] = out[pdb_column].map(
        lambda p: ", ".join(info.get(str(p).upper(), {}).get("uniprot_ids", [])) or "N/A")
    out["Protein_Names"] = out[pdb_column].map(
        lambda p: " | ".join(info.get(str(p).upper(), {}).get("protein_names", [])) or "N/A")
    return out


def parse_uniprot_entry(data: dict) -> dict:
    """Extract protein name, genes and disease comments from a UniProtKB JSON entry."""
    diseases = []
    for comment in data.get("comments") or []:
        if comment.get("commentType") != "DISEASE":
            continue
        d = comment.get("disease") or {}
        if d.get("diseaseId"):
            diseases.append({
                "name": d.get("diseaseId", ""),
                "acronym": d.get("acronym", ""),
                "description": d.get("description", ""),
                "mim": (d.get("diseaseCrossReference") or {}).get("id", ""),
            })
    protein = (((data.get("proteinDescription") or {}).get("recommendedName") or {})
               .get("fullName") or {}).get("value", "")
    genes = [g.get("geneName", {}).get("value", "") for g in data.get("genes") or []
             if g.get("geneName")]
    return {"protein_name": protein, "genes": genes, "diseases": diseases}


def fetch_uniprot_diseases(accessions: Iterable[str], workers: int = 4,
                           session_factory: Callable[[], requests.Session] = get_session,
                           ) -> dict[str, dict]:
    """Fetch UniProtKB entries in parallel and return ``parse_uniprot_entry``
    output per accession (with ``"status"`` set to ``ok`` or an error)."""
    accs = list(dict.fromkeys(a.strip().upper() for a in accessions if a and a.strip()))

    def one(acc: str) -> tuple[str, dict]:
        try:
            r = session_factory().get(UNIPROT_ENTRY_URL.format(accession=acc),
                                      timeout=DEFAULT_TIMEOUT)
            if r.status_code == 404:
                return acc, {"protein_name": "", "genes": [], "diseases": [], "status": "not_found"}
            r.raise_for_status()
            return acc, {**parse_uniprot_entry(r.json()), "status": "ok"}
        except (requests.RequestException, ValueError) as exc:
            return acc, {"protein_name": "", "genes": [], "diseases": [], "status": f"error: {exc}"}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return dict(pool.map(one, accs))


def split_ids(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [v.strip() for v in str(value).split(",") if v.strip() and v.strip() != "N/A"]


def add_disease_annotations(df: pd.DataFrame, uniprot_column: str = "UniProt_IDs") -> pd.DataFrame:
    """Return a copy of ``df`` with ``Disease_Associations`` (``" | "``-joined
    disease names) for the comma-separated accessions in ``uniprot_column``."""
    out = df.copy()
    accs = sorted({a for v in out[uniprot_column] for a in split_ids(v)})
    entries = fetch_uniprot_diseases(accs) if accs else {}

    def diseases_for(value) -> str:
        names = []
        for acc in split_ids(value):
            for d in entries.get(acc.upper(), {}).get("diseases", []):
                if d["name"] not in names:
                    names.append(d["name"])
        return " | ".join(names) if names else "None reported"

    out["Disease_Associations"] = out[uniprot_column].map(diseases_for)
    return out


def matches_keywords(text: str, keywords: Iterable[str]) -> bool:
    text = str(text or "").lower()
    return any(k.strip().lower() in text for k in keywords if k and k.strip())
