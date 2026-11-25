"""Target Search page: UniProt IDs + query molecule -> ranked ligands from live PDB data."""

import streamlit as st

from trackmypdb.similarity import rank_by_similarity
from ui.molecules import hit_cards, query_preview
from ui.viewer3d import structure_viewer
from ui.style import how_to, page_header
from ui.common import (download_csv, protein_info_step, results_table, run_extraction,
                       search_settings, similarity_charts, similarity_summary, smiles_feedback,
                       smiles_lines)
from ui.extraction import uniprot_input


def _load_example() -> None:
    st.session_state["pipe_ids"] = "P37231"
    st.session_state["pipe_smiles"] = "CN(CCOc1ccc(CC2SC(=O)NC2=O)cc1)c1ccccn1"


def render() -> None:
    page_header("Target Search", "Start from your proteins: fetch their live PDB structures "
                "and rank the bound ligands against your molecule.", icon="🎯")
    how_to([
        "Enter UniProt accessions (e.g. `P37231`), or click **Try an example**.",
        "Paste your molecule's SMILES. The preview shows its structure.",
        "Set **Max structures per protein**: 10 is quick, 0 uses all structures (slower).",
        "Click **Run search**, then explore the cards, the 3D view and the table.",
    ], tip="Needs internet. About 1 minute for 10 structures.")
    st.button("Try an example: PPAR-γ + rosiglitazone", icon=":material/auto_awesome:",
              on_click=_load_example)

    ids, limit = uniprot_input("pipe")
    left, right = st.columns([3, 2], vertical_alignment="top")
    with left:
        query = st.text_input("Query SMILES", key="pipe_smiles",
                              placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O (aspirin)")
        valid = smiles_feedback(smiles_lines(query))
        settings = search_settings("pipe", top_n=50, min_similarity=0.0)
    with right:
        query_preview(valid[0] if valid else None)

    if st.button("Run search", type="primary", icon=":material/rocket_launch:",
                 disabled=not (ids and valid)):
        result = run_extraction(ids, limit)
        st.session_state["ligands"] = result.records
        hits = rank_by_similarity(valid[0], result.records, **settings)
        st.session_state["pipeline_hits"] = hits
        st.session_state["pipeline_query"] = valid[0]
        st.session_state.pop("pipeline_hits_proteins", None)

    hits = st.session_state.get("pipeline_hits")
    if hits is None:
        return
    if hits.empty:
        st.info("No ligands with SMILES reached the similarity threshold.")
        return
    similarity_summary(hits)
    hit_cards(hits, n=11, query_smiles=st.session_state.get("pipeline_query"))
    results_table(hits)
    c1, c2 = st.columns(2)
    with c1:
        download_csv(hits, "Download ranked ligands (CSV)", "pipeline_hits")
    with c2:
        download_csv(st.session_state["ligands"], "Download all extracted ligands (CSV)",
                     "pipeline_ligands")
    structure_viewer(hits, "pipe")
    similarity_charts(hits)
    st.divider()
    protein_info_step(hits, "pipeline_hits_proteins")
