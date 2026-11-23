"""Database search page: query the bundled PDB ligand database."""

import pandas as pd
import streamlit as st

from trackmypdb.similarity import rank_by_similarity
from ui.common import (download_csv, ligand_database, protein_info_step, results_table,
                       search_settings, similarity_charts, similarity_summary, smiles_feedback,
                       smiles_lines)


def render() -> None:
    st.header("Database search")
    st.caption("Find PDB ligands similar to your molecule(s) in the bundled ligand "
               "database. No UniProt ID or network access needed.")

    text = st.text_area("Query SMILES (one per line)", height=110,
                        placeholder="CC(=O)Oc1ccccc1C(=O)O\nCn1cnc2c1c(=O)n(C)c(=O)n2C")
    queries = smiles_feedback(smiles_lines(text))
    settings = search_settings("db", top_n=20, min_similarity=0.3)
    db, fps = ligand_database(settings["radius"], settings["n_bits"])
    st.caption(f"Database: {len(db):,} ligand records with SMILES · "
               f"{db['PDB_ID'].nunique():,} PDB entries · "
               f"{db['Heteroatom_Code'].nunique():,} distinct ligands")

    if st.button("Search", type="primary", disabled=not queries):
        frames = []
        for q in queries:
            hits = rank_by_similarity(q, db, precomputed_fps=fps, **settings)
            if len(queries) > 1:
                hits.insert(0, "Query_SMILES", q)
            frames.append(hits)
        hits = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if len(queries) > 1 and not hits.empty:
            hits = hits.sort_values("Tanimoto_Similarity", ascending=False, kind="mergesort")
        st.session_state["db_hits"] = hits.reset_index(drop=True)
        st.session_state.pop("db_hits_proteins", None)

    hits = st.session_state.get("db_hits")
    if hits is None:
        return
    if hits.empty:
        st.info("No ligands reached the similarity threshold. Try lowering it.")
        return
    similarity_summary(hits)
    results_table(hits)
    download_csv(hits, "Download results (CSV)", "database_search")
    similarity_charts(hits)
    st.divider()
    protein_info_step(hits, "db_hits_proteins")
