"""
Substructure search and maximum common substructure (RDKit).

* ``substructure_search`` finds every ligand containing a scaffold given as
  SMILES or SMARTS.
* ``maximum_common_substructure`` returns the largest shared fragment of two
  molecules as SMARTS, used to show *why* two ligands are similar.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS

from .similarity import InvalidSmilesError, bulk_tanimoto, fingerprint, mol_from_smiles

SIMILARITY_COLUMN = "Tanimoto_Similarity"


def parse_query(pattern: str):
    """Parse a substructure query. SMILES is tried first (so ``c1ccccc1`` and
    ``C(=O)O`` behave as chemists expect), then SMARTS for queries such as
    ``[#6]-[#7;H1]``. Returns ``None`` if neither parses."""
    pattern = (pattern or "").strip()
    if not pattern:
        return None
    return Chem.MolFromSmiles(pattern) or Chem.MolFromSmarts(pattern)


def substructure_search(pattern: str, ligands: pd.DataFrame, smiles_column: str = "SMILES",
                        max_results: int | None = None, mols: Sequence | None = None,
                        radius: int = 2, n_bits: int = 2048) -> pd.DataFrame:
    """Rows of ``ligands`` whose molecule contains ``pattern``.

    Adds ``Heavy_Atoms`` and, when the pattern is a valid SMILES, the
    ``Tanimoto_Similarity`` of each match to it. Matches are sorted by
    similarity (most similar first) or, for SMARTS-only patterns, by size
    (smallest first, i.e. closest to the bare scaffold).

    ``mols`` (one RDKit molecule or ``None`` per row) avoids re-parsing the
    same library on every query.
    """
    query = parse_query(pattern)
    if query is None:
        raise InvalidSmilesError(f"Invalid SMILES/SMARTS query: {pattern!r}")
    df = ligands.reset_index(drop=True)
    if df.empty or smiles_column not in df.columns:
        return df.iloc[0:0].assign(Heavy_Atoms=pd.Series(dtype=int))
    if mols is None:
        mols = [mol_from_smiles(s) for s in df[smiles_column]]
    if len(mols) != len(df):
        raise ValueError("mols must have one entry per ligand row")

    params = Chem.SubstructMatchParameters()
    params.useChirality = False
    hit_idx = [i for i, m in enumerate(mols)
               if m is not None and m.HasSubstructMatch(query, params)]
    out = df.iloc[hit_idx].copy()
    out["Heavy_Atoms"] = [mols[i].GetNumHeavyAtoms() for i in hit_idx]

    query_fp = fingerprint(pattern, radius, n_bits) if Chem.MolFromSmiles(pattern) else None
    if query_fp is not None:
        fps = [fingerprint(s, radius, n_bits) for s in out[smiles_column]]
        out[SIMILARITY_COLUMN] = bulk_tanimoto(query_fp, fps) if fps else np.array([])
        out = out.sort_values([SIMILARITY_COLUMN, "Heavy_Atoms"], ascending=[False, True],
                              kind="mergesort")
    else:
        out = out.sort_values("Heavy_Atoms", kind="mergesort")
    if max_results:
        out = out.head(max_results)
    return out.reset_index(drop=True)


def maximum_common_substructure(smiles_a: str, smiles_b: str, timeout: float = 1.0,
                                min_atoms: int = 3) -> str | None:
    """SMARTS of the maximum common substructure of two molecules.

    Rings are matched only to complete rings, so the result is a chemically
    meaningful fragment. Returns ``None`` if either molecule is invalid, no
    fragment with at least ``min_atoms`` atoms is found, or the search times out
    without a result.
    """
    a, b = mol_from_smiles(smiles_a), mol_from_smiles(smiles_b)
    if a is None or b is None:
        return None
    res = rdFMCS.FindMCS([a, b], timeout=max(1, int(round(timeout))),
                         ringMatchesRingOnly=True, completeRingsOnly=True,
                         atomCompare=rdFMCS.AtomCompare.CompareElements,
                         bondCompare=rdFMCS.BondCompare.CompareOrder)
    if not res.smartsString or res.numAtoms < min_atoms:
        return None
    return res.smartsString


def match_atoms(smiles: str, smarts: str) -> tuple[list[int], list[int]]:
    """Atom and bond indices of ``smiles`` matched by a SMILES/SMARTS pattern
    (for highlighting)."""
    mol = mol_from_smiles(smiles)
    patt = parse_query(smarts) if smarts else None
    if mol is None or patt is None:
        return [], []
    atoms = list(mol.GetSubstructMatch(patt))
    aset = set(atoms)
    bonds = [b.GetIdx() for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in aset and b.GetEndAtomIdx() in aset]
    return atoms, bonds
