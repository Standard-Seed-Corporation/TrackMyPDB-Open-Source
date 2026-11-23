"""Similarity ranking page: rank extracted (or uploaded) ligands against a query."""

import pandas as pd
import streamlit as st

from trackmypdb.similarity import InvalidSmilesError, rank_by_similarity
from ui.molecules import hit_cards, query_preview
from ui.viewer3d import structure_viewer
from ui.style import page_header
from ui.common import (PAGES, download_csv, protein_info_step, results_table, search_settings,
                       similarity_charts, similarity_summary, smiles_feedback, smiles_lines)


def _ligand_source() -> pd.DataFrame | None:
    options = ["Upload a CSV"]
    if st.session_state.get("ligands") is not None and not st.session_state["ligands"].empty:
        options.insert(0, "Ligands from the extraction page")
    source = st.radio("Ligands to rank", options, horizontal=True)
    if source.startswith("Ligands from"):
        df = st.session_state["ligands"]
        st.caption(f"{len(df)} ligand records from {df['PDB_ID'].nunique()} PDB entries.")
        return df
    upload = st.file_uploader("CSV with a SMILES column (e.g. an earlier TrackMyPDB export)",
                              type=["csv"])
    if upload is None:
        if len(options) == 1:
            st.page_link(PAGES["extraction"], label="Or extract ligands first",
                         icon=":material/arrow_back:")
        return None
    df = pd.read_csv(upload, dtype=str, keep_default_na=False)
    if "SMILES" not in df.columns:
        st.error("The file needs a column named SMILES.")
        return None
    for col in ("PDB_ID", "Heteroatom_Code"):
        if col not in df.columns:
            df[col] = ""
    return df


def render() -> None:
    page_header("Similarity Ranking", "Rank ligands against your molecule.", icon="⚖️")

    ligands = _ligand_source()
    left, right = st.columns([3, 2], vertical_alignment="top")
    with left:
        query = st.text_input("Query SMILES",
                              placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O (aspirin)")
        valid = smiles_feedback(smiles_lines(query))
        settings = search_settings("sim", top_n=50, min_similarity=0.2)
    with right:
        query_preview(valid[0] if valid else None)

    if st.button("Rank ligands", type="primary", icon=":material/sort:",
                 disabled=ligands is None or not valid):
        try:
            hits = rank_by_similarity(valid[0], ligands, **settings)
        except InvalidSmilesError as exc:
            st.error(str(exc))
            return
        st.session_state["similarity_hits"] = hits
        st.session_state["similarity_query"] = valid[0]
        st.session_state.pop("similarity_hits_proteins", None)

    hits = st.session_state.get("similarity_hits")
    if hits is None:
        return
    if hits.empty:
        st.info("No ligands reached the similarity threshold. Try lowering it.")
        return
    similarity_summary(hits)
    hit_cards(hits, n=11, query_smiles=st.session_state.get("similarity_query"))
    results_table(hits)
    download_csv(hits, "Download ranked ligands (CSV)", "similarity")
    structure_viewer(hits, "sim")
    similarity_charts(hits)
    st.divider()
    protein_info_step(hits, "similarity_hits_proteins")
