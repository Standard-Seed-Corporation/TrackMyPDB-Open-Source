"""Shared Streamlit helpers: cached resources, input widgets, result tables,
downloads, charts and the protein-information step used by several pages."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from trackmypdb import __version__
from trackmypdb.extraction import HeteroatomExtractor

ROOT = Path(__file__).resolve().parent.parent
LOGO = ROOT / "ssc.png"

#: Filled in by streamlit_app.py so pages can link to each other.
PAGES: dict = {}

COLUMN_CONFIG = {
    "UniProt_ID": st.column_config.TextColumn("UniProt"),
    "PDB_ID": st.column_config.TextColumn("PDB ID"),
    "Heteroatom_Code": st.column_config.TextColumn("Ligand"),
    "Chemical_Name": st.column_config.TextColumn("Name", width="medium"),
    "SMILES": st.column_config.TextColumn("SMILES", width="medium"),
    "Query_SMILES": st.column_config.TextColumn("Query", width="small"),
    "Tanimoto_Similarity": st.column_config.ProgressColumn(
        "Similarity", help="Tanimoto similarity (0-1, higher is more similar)",
        format="%.3f", min_value=0.0, max_value=1.0),
    "UniProt_IDs": st.column_config.TextColumn("UniProt IDs", width="medium"),
    "Protein_Names": st.column_config.TextColumn("Protein names", width="large"),
    "Disease_Associations": st.column_config.TextColumn("Diseases", width="large"),
}


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #

@st.cache_resource
def shared_extractor() -> HeteroatomExtractor:
    """One extractor per server process, so resolved ligand codes are cached
    across runs and users (the data is public)."""
    return HeteroatomExtractor()


@st.cache_resource(show_spinner="Loading and fingerprinting the ligand database...")
def ligand_database(radius: int, n_bits: int):
    """The bundled ligand DB plus fingerprints, computed once per setting."""
    from trackmypdb.ligand_db import load_ligand_db
    from trackmypdb.similarity import fingerprints
    db = load_ligand_db(with_smiles_only=True)
    return db, fingerprints(db["SMILES"].tolist(), radius, n_bits)


@st.cache_resource(show_spinner="Preparing substructure search...")
def ligand_mols() -> list:
    """Parsed RDKit molecules for the bundled database (same row order as
    ``ligand_database``), cached for substructure search."""
    from trackmypdb.similarity import mol_from_smiles
    db, _ = ligand_database(2, 2048)
    return [mol_from_smiles(s) for s in db["SMILES"]]


@st.cache_data(max_entries=1000)
def smiles_is_valid(smiles: str) -> bool:
    from trackmypdb.similarity import mol_from_smiles
    return mol_from_smiles(smiles) is not None


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def smiles_lines(text: str) -> list[str]:
    """One SMILES per line; anything after whitespace (e.g. a name) is ignored."""
    return [ln.split()[0] for ln in (text or "").splitlines() if ln.strip()]


def smiles_feedback(smiles_list: list[str]) -> list[str]:
    """Show validation feedback and return only the valid SMILES."""
    valid = [s for s in smiles_list if smiles_is_valid(s)]
    invalid = [s for s in smiles_list if s not in valid]
    if smiles_list and not invalid:
        st.caption(f"✅ {len(valid)} valid structure(s)")
    for s in invalid:
        st.warning(f"Not a valid SMILES and will be skipped: `{s}`")
    return valid


def search_settings(key: str, top_n: int = 50, min_similarity: float = 0.2,
                    max_top: int = 500) -> dict:
    """Result-count, threshold and fingerprint settings in an expander."""
    with st.expander("Search settings"):
        c1, c2, c3, c4 = st.columns(4)
        settings = {
            "top_n": c1.number_input("Max results", 1, max_top, top_n, step=10, key=f"{key}_top"),
            "min_similarity": c2.slider("Min similarity", 0.0, 1.0, min_similarity, 0.05,
                                        key=f"{key}_min"),
            "radius": c3.selectbox("Morgan radius", [1, 2, 3], index=1, key=f"{key}_radius"),
            "n_bits": c4.selectbox("Fingerprint bits", [1024, 2048, 4096], index=1,
                                   key=f"{key}_bits"),
        }
    return settings


# --------------------------------------------------------------------------- #
# Running the extractor with live progress
# --------------------------------------------------------------------------- #

def run_extraction(uniprot_ids: list[str], max_structures: int | None):
    """Run the extractor inside an ``st.status`` box and report the outcome."""
    with st.status("Extracting ligands...", expanded=True) as status:
        bar = st.progress(0.0)
        msg = st.empty()

        def progress(frac: float, text: str) -> None:
            bar.progress(frac)
            msg.caption(text)

        result = shared_extractor().extract(uniprot_ids, max_structures=max_structures,
                                            progress=progress)
        s = result.summary
        status.update(
            label=(f"Found {s['records']} ligand records ({s['unique_ligands']} unique) "
                   f"in {s['structures_requested'] - s['structures_failed']} structures"),
            state="complete" if s["records"] else "error", expanded=False)

    if result.invalid_ids:
        st.warning("Not valid UniProt accessions (skipped): " + ", ".join(result.invalid_ids))
    if result.unmapped_ids:
        st.info("No PDB structures found for: " + ", ".join(result.unmapped_ids))
    if result.failed_structures:
        st.warning(f"{len(result.failed_structures)} structure(s) could not be downloaded: "
                   + ", ".join(sorted(result.failed_structures)))
    return result


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def results_table(df: pd.DataFrame, height: int | None = None) -> None:
    st.dataframe(df, hide_index=True, column_config=COLUMN_CONFIG,
                 height=height or min(38 + 35 * max(len(df), 1), 600))


def download_csv(df: pd.DataFrame, label: str, name: str, key: str | None = None) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"),
                       file_name=f"TrackMyPDB_{name}_{timestamp()}.csv", mime="text/csv",
                       key=key, on_click="ignore")


def similarity_summary(hits: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    sim = hits["Tanimoto_Similarity"]
    c1.metric("Hits", len(hits))
    c2.metric("Best", f"{sim.max():.3f}")
    c3.metric("Mean", f"{sim.mean():.3f}")
    c4.metric("PDB entries", hits["PDB_ID"].nunique())


def similarity_charts(hits: pd.DataFrame) -> None:
    import plotly.express as px

    left, right = st.columns(2)
    with left:
        fig = px.histogram(hits, x="Tanimoto_Similarity", nbins=20,
                           labels={"Tanimoto_Similarity": "Tanimoto similarity"},
                           title="Distribution of similarity scores",
                           color_discrete_sequence=["#2E7D32"])
        fig.update_layout(bargap=0.05, yaxis_title="Ligands", margin=dict(t=50, b=10))
        st.plotly_chart(fig)
    with right:
        top = hits.head(20).copy()
        top["Label"] = top["PDB_ID"] + " · " + top["Heteroatom_Code"]
        fig = px.bar(top, x="Tanimoto_Similarity", y="Label", orientation="h",
                     title="Top 20 matches", color_discrete_sequence=["#2E7D32"],
                     labels={"Tanimoto_Similarity": "Tanimoto similarity", "Label": ""})
        fig.update_layout(yaxis={"categoryorder": "total ascending"},
                          xaxis_range=[0, 1], margin=dict(t=50, b=10))
        st.plotly_chart(fig)


def protein_info_step(hits: pd.DataFrame, state_key: str) -> pd.DataFrame | None:
    """Optional "add UniProt IDs and protein names" step for a hits table.

    The annotated table is stored in ``st.session_state[state_key]`` so it
    survives reruns and can be reused on the Disease Enrichment page.
    """
    from trackmypdb.annotations import add_protein_info

    st.subheader(":material/biotech: Protein targets")
    annotated = st.session_state.get(state_key)
    if annotated is not None and len(annotated) != len(hits):
        annotated = None  # results changed since annotation
    if annotated is None:
        if st.button("Add UniProt IDs & protein names", key=f"{state_key}_btn",
                     icon=":material/add:"):
            with st.spinner(f"Querying RCSB for {hits['PDB_ID'].nunique()} PDB entries..."):
                annotated = add_protein_info(hits)
            st.session_state[state_key] = annotated
    if annotated is not None:
        results_table(annotated)
        accs = sorted({a.strip() for v in annotated["UniProt_IDs"] for a in str(v).split(",")
                       if a.strip() and a.strip() != "N/A"})
        st.caption(f"{len(accs)} distinct UniProt accessions: "
                   + ", ".join(f"[{a}](https://www.uniprot.org/uniprotkb/{a})" for a in accs[:30])
                   + (" ..." if len(accs) > 30 else ""))
        download_csv(annotated, "Download with protein information (CSV)",
                     f"{state_key}_proteins", key=f"{state_key}_dl")
    return annotated


CITATION_APA = (
    "Sharif, S., Gamage, A., Kotawalagedara, K., MR, S., & Bodun, D. (2025). "
    f"TrackMyPDB (Version {__version__}) [Computer software]. Standard Seed Corporation. "
    "https://trackmypdbsscai.streamlit.app/")


@st.cache_data
def _logo_b64() -> str:
    import base64
    return base64.b64encode(LOGO.read_bytes()).decode() if LOGO.exists() else ""


def sidebar_branding() -> None:
    if LOGO.exists():
        st.logo(str(LOGO), size="large")
    logo = _logo_b64()
    img = (f'<img src="data:image/png;base64,{logo}" alt="Standard Seed Corporation logo">'
           if logo else "")
    with st.sidebar:
        st.markdown(
            f'<div class="tmp-brand">{img}'
            '<div class="tmp-brand-org">Developed and released by<br>'
            "<b>Standard Seed Corporation</b></div>"
            f'<span class="tmp-brand-ver">TrackMyPDB v{__version__}</span>'
            '<div class="tmp-brand-team">Suliman Sharif · Anu Gamage · Kalana Kotawalagedara'
            " · Sakeer MR · Damilola Bodun</div>"
            '<div class="tmp-brand-lic">MIT License</div></div>',
            unsafe_allow_html=True)
        with st.popover("Cite TrackMyPDB", icon=":material/format_quote:"):
            st.code(CITATION_APA, language="text", wrap_lines=True)
            st.caption("BibTeX: CITATION.cff in the repository. Please also acknowledge "
                       "RCSB PDB, PDBe, UniProt and RDKit.")
