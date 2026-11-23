"""Home page."""

import streamlit as st

from trackmypdb import __version__
from ui.common import PAGES, ligand_database, smiles_is_valid
from ui.molecules import EXAMPLES, hit_cards, mol_svg

CITATION_APA = (
    "Sharif, S., Gamage, A., Kotawalagedara, K., Sha, S., & Bodun, D. (2025). "
    f"TrackMyPDB (Version {__version__}) [Computer software]. Standard Seed Corporation. "
    "https://trackmypdbsscai.streamlit.app/")

# key, icon, title, one-liner
TOOLS = [
    ("search", ":material/manage_search:", "Database Search", "Similar PDB ligands in seconds"),
    ("pipeline", ":material/route:", "Full Pipeline", "Proteins + molecule → ranked hits"),
    ("extraction", ":material/science:", "Ligand Extraction", "Every ligand of your proteins"),
    ("similarity", ":material/compare_arrows:", "Similarity Ranking", "Rank any ligand list"),
    ("visualizer", ":material/hub:", "Molecule Viewer", "Structure + drug-likeness"),
    ("disease", ":material/health_and_safety:", "Disease Enrichment", "Targets → diseases"),
]



def _hero() -> None:
    chips = "".join(f'<span class="tmp-chip">{c}</span>'
                    for c in ["RCSB PDB", "PDBe", "UniProt", "RDKit", "MCP-ready"])
    art = mol_svg(EXAMPLES["Imatinib"], 420, 260, white=True)
    st.markdown(
        '<div class="tmp-hero"><div class="tmp-hero-text"><h1>TrackMyPDB</h1>'
        "<p>Ligands from protein structures, ranked by chemical similarity.</p>"
        f'{chips}</div><div class="tmp-hero-art">{art}</div></div>',
        unsafe_allow_html=True)


def _stats() -> None:
    db, _ = ligand_database(2, 2048)
    items = [("🧬", f"{db['PDB_ID'].nunique():,}", "PDB entries"),
             ("💊", f"{db['Heteroatom_Code'].nunique():,}", "distinct ligands"),
             ("📚", f"{len(db):,}", "ligand records"),
             ("⚡", "< 1 s", "database search")]
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
    st.page_link(PAGES["search"], label="See all matches in Database Search",
                 icon=":material/arrow_forward:")


def _tools() -> None:
    cols = st.columns(3)
    for i, (key, icon, title, line) in enumerate(TOOLS):
        with cols[i % 3].container(border=True):
            st.markdown(f"#### {icon} {title}")
            st.caption(line)
            st.page_link(PAGES[key], label="Open", icon=":material/arrow_forward:")


def render() -> None:
    _hero()
    _stats()
    _quick_try()
    st.markdown("### :material/apps: Tools")
    _tools()

    st.markdown(
        '<div class="tmp-flow">' + '<span class="tmp-arrow">→</span>'.join(
            f'<span class="tmp-step">{s}</span>'
            for s in ["UniProt", "PDB structures", "Ligands", "SMILES", "Similarity",
                      "Targets & diseases"]) + "</div>", unsafe_allow_html=True)

    with st.expander(":material/format_quote: Cite TrackMyPDB"):
        st.code(CITATION_APA, language="text")
        st.caption("BibTeX: see CITATION.cff in the repository. Please also acknowledge "
                   "RCSB PDB, PDBe, UniProt and RDKit.")
    with st.expander(":material/groups: Team"):
        st.markdown("Suliman Sharif (supervisor) · Anu Gamage (lead) · Kalana Kotawalagedara · "
                    "Sakeer Sha · Damilola Bodun  \nStandard Seed Corporation · MIT License")
