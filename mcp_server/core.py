"""
TrackMyPDB tool functions for the MCP server and the in-app agent.

Thin wrappers around the ``trackmypdb`` library that return plain,
JSON-serialisable dictionaries. Result sizes are capped so that tool output
stays small enough for a language-model context window.

MIT License.
"""

from __future__ import annotations

import functools

import pandas as pd

from trackmypdb.extraction import HeteroatomExtractor
from trackmypdb.ligand_db import load_ligand_db

MAX_RECORDS_RETURNED = 200
LIGAND_FIELDS = ["UniProt_ID", "PDB_ID", "Heteroatom_Code", "Chemical_Name", "Formula", "SMILES"]
HIT_FIELDS = ["PDB_ID", "Heteroatom_Code", "Chemical_Name", "Formula", "SMILES",
              "Tanimoto_Similarity"]


def _records(df: pd.DataFrame, fields: list[str], limit: int = MAX_RECORDS_RETURNED) -> list[dict]:
    cols = [c for c in fields if c in df.columns]
    return df[cols].head(limit).to_dict(orient="records")


def get_pdbs_for_uniprot(uniprot_id: str) -> list:
    """PDB structure IDs mapped to a UniProt accession, best first."""
    return HeteroatomExtractor().get_pdbs_for_uniprot((uniprot_id or "").strip())


def extract_heteroatoms(uniprot_ids, max_pdbs_per_uniprot: int = 5) -> dict:
    """Extract ligands and their SMILES from the PDB structures of the given
    UniProt IDs. ``max_pdbs_per_uniprot`` = 0 means no cap."""
    if isinstance(uniprot_ids, str):
        uniprot_ids = [uniprot_ids]
    result = HeteroatomExtractor().extract(uniprot_ids,
                                           max_structures=max_pdbs_per_uniprot or None)
    return {
        **result.summary,
        "pdbs_per_uniprot": result.pdbs_per_uniprot,
        "failed_structures": result.failed_structures,
        "truncated": len(result.records) > MAX_RECORDS_RETURNED,
        "heteroatoms": _records(result.records, LIGAND_FIELDS),
    }


def analyze_similarity(target_smiles: str, heteroatoms, top_n: int = 10,
                       min_similarity: float = 0.0, radius: int = 2, n_bits: int = 2048) -> dict:
    """Rank heteroatom records (dicts with a ``SMILES`` key) by Tanimoto
    similarity to ``target_smiles``."""
    from trackmypdb.similarity import InvalidSmilesError, rank_by_similarity

    df = pd.DataFrame(heteroatoms or [])
    if df.empty or "SMILES" not in df.columns:
        return {"target_smiles": target_smiles, "result_count": 0, "results": [],
                "error": "No heteroatom records with a SMILES field were provided."}
    try:
        hits = rank_by_similarity(target_smiles, df, top_n=top_n, min_similarity=min_similarity,
                                  radius=radius, n_bits=n_bits)
    except InvalidSmilesError as exc:
        return {"target_smiles": target_smiles, "result_count": 0, "results": [], "error": str(exc)}
    hits["Tanimoto_Similarity"] = hits["Tanimoto_Similarity"].round(4)
    return {"target_smiles": target_smiles, "result_count": int(len(hits)),
            "results": _records(hits, HIT_FIELDS)}


def run_pipeline(uniprot_ids, target_smiles: str, max_pdbs_per_uniprot: int = 5,
                 top_n: int = 10, min_similarity: float = 0.0, radius: int = 2,
                 n_bits: int = 2048) -> dict:
    """Extract ligands for UniProt IDs, then rank them against a target molecule."""
    from trackmypdb.similarity import InvalidSmilesError, rank_by_similarity

    if isinstance(uniprot_ids, str):
        uniprot_ids = [uniprot_ids]
    result = HeteroatomExtractor().extract(uniprot_ids,
                                           max_structures=max_pdbs_per_uniprot or None)
    summary = {**result.summary, "pdbs_per_uniprot": result.pdbs_per_uniprot}
    try:
        hits = rank_by_similarity(target_smiles, result.records, top_n=top_n,
                                  min_similarity=min_similarity, radius=radius, n_bits=n_bits)
    except InvalidSmilesError as exc:
        return {"extraction_summary": summary, "similarity": {"error": str(exc), "results": []}}
    hits["Tanimoto_Similarity"] = hits["Tanimoto_Similarity"].round(4)
    return {"extraction_summary": summary,
            "similarity": {"target_smiles": target_smiles, "result_count": int(len(hits)),
                           "results": _records(hits, ["UniProt_ID"] + HIT_FIELDS)}}


@functools.lru_cache(maxsize=4)
def _db_fingerprints(radius: int, n_bits: int):
    from trackmypdb.similarity import fingerprints
    db = load_ligand_db(with_smiles_only=True)
    return db, fingerprints(db["SMILES"].tolist(), radius, n_bits)


def search_ligand_database(target_smiles: str, top_n: int = 10, min_similarity: float = 0.3,
                           radius: int = 2, n_bits: int = 2048) -> dict:
    """Search the bundled offline PDB ligand database by similarity (no network)."""
    from trackmypdb.similarity import InvalidSmilesError, rank_by_similarity

    db, fps = _db_fingerprints(radius, n_bits)
    try:
        hits = rank_by_similarity(target_smiles, db, top_n=top_n, min_similarity=min_similarity,
                                  radius=radius, n_bits=n_bits, precomputed_fps=fps)
    except InvalidSmilesError as exc:
        return {"target_smiles": target_smiles, "result_count": 0, "results": [], "error": str(exc)}
    hits["Tanimoto_Similarity"] = hits["Tanimoto_Similarity"].round(4)
    return {"target_smiles": target_smiles, "result_count": int(len(hits)),
            "results": _records(hits, HIT_FIELDS)}


def compare_smiles(smiles_a: str, smiles_b: str, radius: int = 2, n_bits: int = 2048) -> dict:
    """Tanimoto similarity between two molecules given as SMILES."""
    from trackmypdb.similarity import InvalidSmilesError, compare_smiles as _compare, fingerprint

    try:
        sim = _compare(smiles_a, smiles_b, radius, n_bits)
    except InvalidSmilesError as exc:
        return {"error": str(exc),
                "valid_a": fingerprint(smiles_a, radius, n_bits) is not None,
                "valid_b": fingerprint(smiles_b, radius, n_bits) is not None}
    return {"smiles_a": smiles_a, "smiles_b": smiles_b, "tanimoto_similarity": round(sim, 4)}
