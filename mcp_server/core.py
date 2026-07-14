"""
TrackMyPDB core tool functions (Streamlit-free).

These wrap the existing backend classes so they run anywhere -- the MCP server,
the in-app Claude agent, tests, or a notebook -- and return plain,
JSON-serialisable Python objects instead of rendering Streamlit UI.

MIT License - Open Source Project.
"""

from __future__ import annotations

import sys

# --- Install the Streamlit shim BEFORE importing the backend ------------------
# The backend does `import streamlit as st`. In the MCP server process there is
# no Streamlit runtime, so we register a silent no-op shim. setdefault means:
#   * MCP server process  -> shim is installed (streamlit not imported yet)
#   * inside the live app -> real streamlit is already imported, so this is a
#     no-op and the app keeps working normally.
from . import st_shim as _st_shim
sys.modules.setdefault("streamlit", _st_shim)

from backend.heteroatom_extractor import HeteroatomExtractor  # noqa: E402
from backend.similarity_analyzer import MolecularSimilarityAnalyzer  # noqa: E402

import pandas as pd  # noqa: E402


# ------------------------------------------------------------------ tools -----

def get_pdbs_for_uniprot(uniprot_id: str) -> list:
    """Return the list of PDB structure IDs mapped to a UniProt accession."""
    return HeteroatomExtractor().get_pdbs_for_uniprot((uniprot_id or "").strip())


def extract_heteroatoms(uniprot_ids, max_pdbs_per_uniprot: int = 5,
                        drug_like_only: bool = True) -> dict:
    """
    Extract drug-like heteroatoms (ligands) and their SMILES from the PDB
    structures associated with the given UniProt IDs.

    max_pdbs_per_uniprot caps how many structures are downloaded per UniProt ID
    (public APIs are slow); set to 0 for no cap.
    """
    if isinstance(uniprot_ids, str):
        uniprot_ids = [uniprot_ids]
    uniprot_ids = [u.strip() for u in uniprot_ids if u and u.strip()]

    ex = HeteroatomExtractor()
    records = []
    pdbs_per_uniprot = {}

    for up in uniprot_ids:
        pdbs = ex.get_pdbs_for_uniprot(up)
        if max_pdbs_per_uniprot:
            pdbs = pdbs[:max_pdbs_per_uniprot]
        pdbs_per_uniprot[up] = pdbs
        for pdb in pdbs:
            lines = ex.download_pdb(pdb)
            if not lines:
                continue
            records.extend(ex.process_pdb_heteroatoms(pdb, up, lines))

    if drug_like_only:
        real = [r for r in records
                if r.get("Heteroatom_Code") not in ("NO_DRUG_HETEROATOMS", "NO_HETEROATOMS")]
        records = real or records

    with_smiles = [r for r in records if r.get("SMILES")]
    return {
        "uniprot_ids": uniprot_ids,
        "pdbs_per_uniprot": pdbs_per_uniprot,
        "record_count": len(records),
        "records_with_smiles": len(with_smiles),
        "heteroatoms": records,
    }


def analyze_similarity(target_smiles: str, heteroatoms, top_n: int = 10,
                       min_similarity: float = 0.0, radius: int = 2,
                       n_bits: int = 2048) -> dict:
    """
    Rank the supplied heteroatom records by Tanimoto similarity (Morgan
    fingerprints) against a target SMILES. `heteroatoms` is a list of dicts as
    returned by extract_heteroatoms()['heteroatoms'].
    """
    df = pd.DataFrame(heteroatoms or [])
    if df.empty or "SMILES" not in df.columns:
        return {"target_smiles": target_smiles, "result_count": 0, "results": [],
                "error": "No heteroatom records with a SMILES column were provided."}

    analyzer = MolecularSimilarityAnalyzer(radius=radius, n_bits=n_bits)
    processed = analyzer.load_and_process_dataframe(df)
    if processed.empty:
        return {"target_smiles": target_smiles, "result_count": 0, "results": [],
                "error": "None of the provided SMILES could be parsed into fingerprints."}

    try:
        results = analyzer.find_similar_ligands(
            target_smiles=target_smiles, processed_df=processed,
            top_n=top_n, min_similarity=min_similarity)
    except ValueError as exc:
        return {"target_smiles": target_smiles, "result_count": 0, "results": [],
                "error": str(exc)}

    cols = [c for c in ["PDB_ID", "Heteroatom_Code", "Chemical_Name", "Formula",
                        "SMILES", "Tanimoto_Similarity"] if c in results.columns]
    out = results[cols].copy()
    if "Tanimoto_Similarity" in out.columns:
        out["Tanimoto_Similarity"] = out["Tanimoto_Similarity"].round(4)
    return {"target_smiles": target_smiles, "result_count": int(len(out)),
            "results": out.to_dict(orient="records")}


def run_pipeline(uniprot_ids, target_smiles: str, max_pdbs_per_uniprot: int = 5,
                 top_n: int = 10, min_similarity: float = 0.0, radius: int = 2,
                 n_bits: int = 2048) -> dict:
    """End-to-end: extract heteroatoms for UniProt IDs, then rank them by
    similarity to a target molecule. This is the tool the agent should prefer."""
    extraction = extract_heteroatoms(uniprot_ids, max_pdbs_per_uniprot=max_pdbs_per_uniprot)
    similarity = analyze_similarity(target_smiles, extraction["heteroatoms"],
                                    top_n=top_n, min_similarity=min_similarity,
                                    radius=radius, n_bits=n_bits)
    summary = {k: v for k, v in extraction.items() if k != "heteroatoms"}
    return {"extraction_summary": summary, "similarity": similarity}


def compare_smiles(smiles_a: str, smiles_b: str, radius: int = 2,
                   n_bits: int = 2048) -> dict:
    """Compute Tanimoto similarity between two molecules given as SMILES."""
    analyzer = MolecularSimilarityAnalyzer(radius=radius, n_bits=n_bits)
    fp_a = analyzer.smiles_to_fingerprint(smiles_a)
    fp_b = analyzer.smiles_to_fingerprint(smiles_b)
    if fp_a is None or fp_b is None:
        return {"error": "One or both SMILES are invalid.",
                "valid_a": fp_a is not None, "valid_b": fp_b is not None}
    sim = analyzer.calculate_tanimoto_similarity(fp_a, fp_b)
    return {"smiles_a": smiles_a, "smiles_b": smiles_b,
            "tanimoto_similarity": round(float(sim), 4)}
