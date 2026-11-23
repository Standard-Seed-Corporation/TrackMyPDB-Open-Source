"""Home page."""

import streamlit as st

from trackmypdb import __version__
from ui.common import PAGES

CITATION_APA = (
    "Sharif, S., Gamage, A., Kotawalagedara, K., Sha, S., & Bodun, D. (2025). "
    f"TrackMyPDB (Version {__version__}) [Computer software]. Standard Seed Corporation. "
    "https://trackmypdbsscai.streamlit.app/")

TOOLS = [
    ("Database Search", "Find co-crystallised PDB ligands similar to your molecule in the "
     "bundled ligand database. Fast, offline, no protein needed.", "search"),
    ("Full Pipeline", "UniProt IDs + a query molecule → ligands from every mapped PDB "
     "structure, ranked by similarity, in one step.", "pipeline"),
    ("Ligand Extraction", "Collect all ligands (with SMILES) bound in the PDB structures "
     "of your proteins.", "extraction"),
    ("Similarity Ranking", "Rank extracted or uploaded ligands against a query SMILES "
     "using Morgan fingerprints and Tanimoto similarity.", "similarity"),
    ("Molecule Viewer", "Draw a structure and compute drug-likeness descriptors.", "visualizer"),
    ("Disease Enrichment", "Annotate protein targets with UniProt disease involvement "
     "and filter by disease area.", "disease"),
]


def render() -> None:
    st.title("TrackMyPDB")
    st.markdown("##### Ligands from protein structures, ranked by chemical similarity")
    st.write(
        "TrackMyPDB finds the small molecules that have been co-crystallised with your "
        "proteins of interest in the Protein Data Bank and ranks them by fingerprint "
        "similarity to a query compound, linking ligand chemistry back to protein "
        "targets and their disease associations.")

    cols = st.columns(3)
    for i, (title, text, key) in enumerate(TOOLS):
        with cols[i % 3].container(border=True, height=190):
            st.markdown(f"**{title}**")
            st.caption(text)
            if key in PAGES:
                st.page_link(PAGES[key], label=f"Open {title}", icon=":material/arrow_forward:")

    st.divider()
    tab1, tab2, tab3 = st.tabs(["How it works", "Cite", "Team"])
    with tab1:
        st.markdown(
            "1. **Map** UniProt accessions to PDB entries (PDBe best-structure mapping).\n"
            "2. **Extract** HETATM components from each structure (mmCIF), dropping water, "
            "ions, buffers and modified residues.\n"
            "3. **Resolve** each component once to SMILES via the RCSB Chemical Component "
            "Dictionary, with an offline fallback database.\n"
            "4. **Rank** ligands by Tanimoto similarity of Morgan fingerprints (RDKit).\n"
            "5. **Annotate** hits with UniProt accessions, protein names and diseases.\n\n"
            "Everything shown here is also available as a Python library, a command-line "
            "tool (`trackmypdb --help`) and an MCP server for AI assistants.")
    with tab2:
        st.write("If you use TrackMyPDB in your work, please cite it:")
        st.code(CITATION_APA, language="text")
        st.caption("A `CITATION.cff` file in the repository provides BibTeX and other formats. "
                   "Please also acknowledge the RCSB PDB, PDBe, UniProt and RDKit.")
    with tab3:
        st.markdown(
            "**Project supervisor:** Suliman Sharif  \n"
            "**Lead engineer:** Anu Gamage  \n"
            "**Associate engineers:** Kalana Kotawalagedara, Sakeer Sha, Damilola Bodun  \n\n"
            "Developed and released by Standard Seed Corporation under the MIT License.")
