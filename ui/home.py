"""Home page."""

import streamlit as st

from ui.common import PAGES, ligand_database, smiles_is_valid
from ui.molecules import EXAMPLES, hit_cards, mol_svg


def _hero() -> None:
    steps = ["UniProt", "PDB structures", "Ligands", "Similarity", "Diseases"]
    flow = '<span class="tmp-hero-arrow">→</span>'.join(
        f'<span class="tmp-chip">{s}</span>' for s in steps)
    art = mol_svg(EXAMPLES["Imatinib"], 420, 260, white=True)
    st.markdown(
        '<div class="tmp-hero"><div class="tmp-hero-text"><h1>TrackMyPDB</h1>'
        "<p>Ligands from protein structures, ranked by chemical similarity.</p>"
        f'<div class="tmp-hero-flow">{flow}</div></div>'
        f'<div class="tmp-hero-art">{art}</div></div>',
        unsafe_allow_html=True)


def _stats() -> None:
    db, _ = ligand_database(2, 2048)
    items = [("🧬", f"{db['PDB_ID'].nunique():,}", "PDB entries"),
             ("💊", f"{db['Heteroatom_Code'].nunique():,}", "distinct ligands"),
             ("📚", f"{len(db):,}", "ligand records"),
             ("⚡", "< 1 s", "Quick Search")]
    st.markdown('<div class="tmp-stats">' + "".join(
        f'<div class="tmp-stat" style="animation-delay:{i * 0.08:.2f}s"><i>{icon}</i>'
        f'<b>{v}</b><span>{k}</span></div>' for i, (icon, v, k) in enumerate(items))
        + "</div>", unsafe_allow_html=True)


def _use_example() -> None:
    choice = st.session_state.get("home_example")
    if choice:
        st.session_state["home_smiles"] = EXAMPLES[choice]


def _quick_try() -> None:
    from trackmypdb.similarity import rank_by_similarity

    st.markdown("### :material/bolt: Try it")
    st.session_state.setdefault("home_smiles", EXAMPLES["Imatinib"])
    st.pills("Examples", list(EXAMPLES), key="home_example", on_change=_use_example,
             label_visibility="collapsed")
    smiles = st.text_input("SMILES", key="home_smiles", label_visibility="collapsed",
                           placeholder="Paste any SMILES").strip()
    if not smiles:
        return
    if not smiles_is_valid(smiles):
        st.caption("Not a valid SMILES.")
        return
    db, fps = ligand_database(2, 2048)
    hits = rank_by_similarity(smiles, db, top_n=5, precomputed_fps=fps)
    hit_cards(hits, n=5, query_smiles=smiles)
    st.page_link(PAGES["search"], label="See all matches in Quick Search",
                 icon=":material/arrow_forward:")


def render() -> None:
    _hero()
    _stats()
    _quick_try()
    st.markdown('<p class="tmp-powered">Powered by open data and software from '
                "RCSB PDB · PDBe · UniProt · RDKit</p>", unsafe_allow_html=True)
