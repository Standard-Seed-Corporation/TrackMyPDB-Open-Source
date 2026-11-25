"""Quick Search page: similarity or substructure search in the built-in ligand library."""

import pandas as pd
import streamlit as st

from trackmypdb.similarity import rank_by_similarity
from trackmypdb.substructure import parse_query, substructure_search
from ui.common import (download_csv, ligand_database, ligand_mols, protein_info_step,
                       results_table, search_settings, similarity_charts, similarity_summary,
                       smiles_feedback, smiles_lines)
from ui.molecules import EXAMPLES, hit_cards, query_preview
from ui.style import how_to, page_header
from ui.viewer3d import structure_viewer

SCAFFOLDS = {
    "Benzimidazole": "c1ccc2[nH]cnc2c1",
    "Quinazoline": "c1ccc2ncncc2c1",
    "Sulfonamide": "S(=O)(=O)N",
    "Piperazine": "C1CNCCN1",
    "Steroid core": "C1CCC2C(C1)CCC3C2CCC4CCCC34",
    "Carboxylic acid": "[CX3](=O)[OX2H1]",
}


def _use_example(mode: str) -> None:
    choice = st.session_state.get(f"db_ex_{mode}")
    if choice:
        examples = SCAFFOLDS if mode == "Substructure" else EXAMPLES
        st.session_state[f"db_text_{mode}"] = examples[choice]


def _run_similarity(queries, db, fps, settings) -> pd.DataFrame:
    frames = []
    for q in queries:
        hits = rank_by_similarity(q, db, precomputed_fps=fps, **settings)
        if len(queries) > 1:
            hits.insert(0, "Query_SMILES", q)
        frames.append(hits)
    hits = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(queries) > 1 and not hits.empty:
        hits = hits.sort_values("Tanimoto_Similarity", ascending=False, kind="mergesort")
    return hits.reset_index(drop=True)


def render() -> None:
    page_header("Quick Search", "Search the built-in PDB ligand library. Instant, "
                "no protein needed.", icon="🔎")
    how_to([
        "Choose **Similarity** (molecules like yours) or **Substructure** "
        "(ligands containing a scaffold).",
        "Click an example or paste a SMILES. Substructure also accepts SMARTS.",
        "Click **Search**.",
        "Explore the cards, pick a hit for the **3D view**, download the CSV or add "
        "protein names.",
    ], tip="Searches the built-in library, with no external database calls, so it is "
           "instant. It covers about 2,600 PDB entries; for complete, up-to-date results on "
           "a specific protein, use Target Search.")

    mode = st.segmented_control("Search type", ["Similarity", "Substructure"],
                                default="Similarity", key="db_mode") or "Similarity"
    sub = mode == "Substructure"
    st.pills("Examples", list(SCAFFOLDS if sub else EXAMPLES), key=f"db_ex_{mode}",
             on_change=_use_example, args=(mode,))

    left, right = st.columns([3, 2], vertical_alignment="top")
    with left:
        if sub:
            text = st.text_input("Scaffold (SMILES or SMARTS)", key="db_text_Substructure",
                                 placeholder="e.g. c1ccc2ncncc2c1 (quinazoline)")
            pattern = text.strip()
            ok = bool(pattern) and parse_query(pattern) is not None
            if pattern and not ok:
                st.warning("Not a valid SMILES or SMARTS pattern.")
            max_results = st.number_input("Max results", 1, 1000, 50, step=10, key="db_sub_top")
            queries = [pattern] if ok else []
        else:
            text = st.text_area("Query SMILES (one per line)", height=110, key="db_text_Similarity",
                                placeholder="CC(=O)Oc1ccccc1C(=O)O\nCn1cnc2c1c(=O)n(C)c(=O)n2C")
            queries = smiles_feedback(smiles_lines(text))
            settings = search_settings("db", top_n=20, min_similarity=0.3)
    with right:
        query_preview(queries[0] if queries else None)

    radius, n_bits = (2, 2048) if sub else (settings["radius"], settings["n_bits"])
    db, fps = ligand_database(radius, n_bits)
    st.caption(f"{len(db):,} ligand records · {db['PDB_ID'].nunique():,} PDB entries · "
               f"{db['Heteroatom_Code'].nunique():,} distinct ligands")

    if st.button("Search", type="primary", icon=":material/search:", disabled=not queries):
        if sub:
            hits = substructure_search(queries[0], db, max_results=max_results, mols=ligand_mols())
        else:
            hits = _run_similarity(queries, db, fps, settings)
        st.session_state["db_hits"] = hits
        st.session_state["db_query"] = (mode, queries[0])
        st.session_state.pop("db_hits_proteins", None)

    hits = st.session_state.get("db_hits")
    if hits is None:
        return
    if hits.empty:
        st.info("No matching ligands. Try a smaller scaffold or a lower similarity threshold.")
        return
    run_mode, query = st.session_state.get("db_query", ("Similarity", None))

    if run_mode == "Substructure":
        c1, c2, c3 = st.columns(3)
        c1.metric("Matches", len(hits))
        c2.metric("PDB entries", hits["PDB_ID"].nunique())
        c3.metric("Distinct ligands", hits["Heteroatom_Code"].nunique())
        hit_cards(hits, n=11, query_smiles=None, highlight=query)
    else:
        similarity_summary(hits)
        hit_cards(hits, n=11, query_smiles=query)
    results_table(hits)
    download_csv(hits, "Download results (CSV)", "database_search")
    structure_viewer(hits, "db")
    if "Tanimoto_Similarity" in hits.columns and hits["Tanimoto_Similarity"].notna().all():
        similarity_charts(hits)
    st.divider()
    protein_info_step(hits, "db_hits_proteins")
