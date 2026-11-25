"""Ligand extraction page."""

import streamlit as st

from trackmypdb.extraction import parse_uniprot_ids
from ui.style import how_to, page_header
from ui.common import PAGES, download_csv, results_table, run_extraction
from ui.viewer3d import structure_viewer

EXAMPLE_IDS = "P37231, Q9UNQ0, P06276"


def uniprot_input(key: str) -> tuple[list[str], int | None]:
    text = st.text_area("UniProt accessions", key=f"{key}_ids", height=90,
                        placeholder=f"Comma, space or newline separated, e.g. {EXAMPLE_IDS}")
    c1, c2 = st.columns([1, 2])
    limit = c1.number_input("Max structures per protein", 0, 1000, 10, step=5,
                            key=f"{key}_max",
                            help="Best-ranked structures are used first. 0 = all structures "
                                 "(can take a long time for well-studied proteins).")
    ids = parse_uniprot_ids(text)
    if ids:
        c2.caption(f"{len(ids)} accession(s): {', '.join(ids)}")
    return ids, (limit or None)


def _load_example() -> None:
    st.session_state["extract_ids"] = "P37231, P06276"


def render() -> None:
    page_header("Ligand Extraction", "List every ligand bound in the PDB structures of your "
                "proteins, with SMILES.", icon="🔬")
    how_to([
        "Enter UniProt accessions, or click **Try an example**.",
        "Choose how many structures per protein to use (best structures first).",
        "Click **Extract ligands**.",
        "Download the table, view a ligand in 3D, or rank them in **Similarity Ranking**.",
    ], tip="Water, ions, buffers and modified residues are removed automatically.")
    st.button("Try an example: PPAR-γ and butyrylcholinesterase", icon=":material/auto_awesome:",
              on_click=_load_example)

    ids, limit = uniprot_input("extract")
    if st.button("Extract ligands", type="primary", icon=":material/science:", disabled=not ids):
        result = run_extraction(ids, limit)
        st.session_state["ligands"] = result.records

    ligands = st.session_state.get("ligands")
    if ligands is None:
        return
    if ligands.empty:
        st.info("No ligands found. Check the accessions or raise the structure limit.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", len(ligands))
    c2.metric("PDB entries", ligands["PDB_ID"].nunique())
    c3.metric("Unique ligands", ligands["Heteroatom_Code"].nunique())
    c4.metric("With SMILES", int((ligands["SMILES"] != "").sum()))
    results_table(ligands)
    structure_viewer(ligands[ligands["SMILES"] != ""], "extract", n=200)
    c1, c2 = st.columns(2)
    with c1:
        download_csv(ligands, "Download ligands (CSV)", "ligands")
    with c2:
        st.page_link(PAGES["similarity"], label="Rank these ligands against a molecule",
                     icon=":material/arrow_forward:")
