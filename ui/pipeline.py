"""Full pipeline page: UniProt IDs + query molecule -> ranked ligands."""

import streamlit as st

from trackmypdb.similarity import rank_by_similarity
from ui.common import (download_csv, protein_info_step, results_table, run_extraction,
                       search_settings, similarity_charts, similarity_summary, smiles_feedback,
                       smiles_lines)
from ui.extraction import uniprot_input


def render() -> None:
    st.header("Full pipeline")
    st.caption("Extract the ligands of your proteins' PDB structures and rank them "
               "against a query molecule in one step.")

    ids, limit = uniprot_input("pipe")
    query = st.text_input("Query SMILES", key="pipe_smiles",
                          placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O (aspirin)")
    valid = smiles_feedback(smiles_lines(query))
    settings = search_settings("pipe", top_n=50, min_similarity=0.0)

    if st.button("Run pipeline", type="primary", disabled=not (ids and valid)):
        result = run_extraction(ids, limit)
        st.session_state["ligands"] = result.records
        hits = rank_by_similarity(valid[0], result.records, **settings)
        st.session_state["pipeline_hits"] = hits
        st.session_state.pop("pipeline_hits_proteins", None)

    hits = st.session_state.get("pipeline_hits")
    if hits is None:
        return
    if hits.empty:
        st.info("No ligands with SMILES reached the similarity threshold.")
        return
    similarity_summary(hits)
    results_table(hits)
    c1, c2 = st.columns(2)
    with c1:
        download_csv(hits, "Download ranked ligands (CSV)", "pipeline_hits")
    with c2:
        download_csv(st.session_state["ligands"], "Download all extracted ligands (CSV)",
                     "pipeline_ligands")
    similarity_charts(hits)
    st.divider()
    protein_info_step(hits, "pipeline_hits_proteins")
