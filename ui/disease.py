"""Disease enrichment page: UniProt disease involvement for protein targets."""

import pandas as pd
import streamlit as st

from trackmypdb.annotations import (DISEASE_CATEGORIES, add_disease_annotations,
                                    add_protein_info, matches_keywords)
from trackmypdb.extraction import is_valid_uniprot_accession, parse_uniprot_ids
from ui.style import page_header
from ui.common import download_csv, results_table

SOURCES = {
    "Extracted ligands": "ligands",
    "Similarity ranking results": "similarity_hits",
    "Database search results": "db_hits",
    "Full pipeline results": "pipeline_hits",
}


def _with_uniprot_column(df: pd.DataFrame, key: str) -> pd.DataFrame | None:
    """Return the table with a ``UniProt_IDs`` column, annotating it if needed."""
    if "UniProt_IDs" in df.columns:
        return df
    if "UniProt_ID" in df.columns:  # extraction output already knows its protein
        return df.assign(UniProt_IDs=df["UniProt_ID"])
    cached = st.session_state.get(f"{key}_proteins")
    if cached is not None and len(cached) == len(df):
        return cached
    st.caption("These results do not have protein accessions yet.")
    if st.button("Look up proteins for these PDB entries"):
        with st.spinner("Querying RCSB..."):
            cached = add_protein_info(df)
        st.session_state[f"{key}_proteins"] = cached
        return cached
    return None


def _choose_input() -> tuple[pd.DataFrame | None, str]:
    available = {label: key for label, key in SOURCES.items()
                 if st.session_state.get(key) is not None and not st.session_state[key].empty}
    label = st.radio("Proteins to annotate", list(available) + ["Enter UniProt IDs"],
                     horizontal=True)
    if label == "Enter UniProt IDs":
        ids = parse_uniprot_ids(st.text_area("UniProt accessions", height=90,
                                             placeholder="P04637, P37231, P00533"))
        bad = [i for i in ids if not is_valid_uniprot_accession(i)]
        if bad:
            st.warning("Skipping invalid accessions: " + ", ".join(bad))
        ids = [i for i in ids if i not in bad]
        return (pd.DataFrame({"UniProt_IDs": ids}) if ids else None), "manual:" + ",".join(ids)
    key = available[label]
    return _with_uniprot_column(st.session_state[key], key), key


def render() -> None:
    page_header("Disease Enrichment", "Which diseases involve your targets?", icon="🩺")

    df, source_id = _choose_input()
    if df is None or df.empty:
        return

    if st.button("Fetch disease annotations", type="primary"):
        with st.spinner(f"Querying UniProt for {df['UniProt_IDs'].nunique()} protein(s)..."):
            st.session_state["disease_table"] = (source_id, add_disease_annotations(df))

    stored = st.session_state.get("disease_table")
    if not stored or stored[0] != source_id:
        return
    table = stored[1]

    c1, c2 = st.columns(2)
    cats = c1.multiselect("Disease areas", list(DISEASE_CATEGORIES))
    custom = c2.text_input("Extra keywords (comma separated)", placeholder="leukemia, obesity")
    keywords = [k for c in cats for k in DISEASE_CATEGORIES[c]]
    keywords += [k.strip() for k in custom.split(",") if k.strip()]
    shown = table
    if keywords:
        shown = table[table["Disease_Associations"].map(lambda t: matches_keywords(t, keywords))]
        st.caption(f"{len(shown)} of {len(table)} rows match the selected disease areas.")

    results_table(shown)
    download_csv(shown, "Download (CSV)", "disease_enrichment")

    names = [d.strip() for v in shown["Disease_Associations"] for d in str(v).split("|")
             if d.strip() and d.strip() != "None reported"]
    if names:
        import plotly.express as px
        counts = pd.Series(names).value_counts().head(20)
        fig = px.bar(x=counts.values, y=counts.index, orientation="h",
                     labels={"x": "Rows", "y": ""}, title="Most frequent disease associations",
                     color_discrete_sequence=["#C62828"])
        fig.update_layout(yaxis={"categoryorder": "total ascending"},
                          height=max(300, 28 * len(counts) + 100))
        st.plotly_chart(fig)
    else:
        st.info("No disease associations recorded in UniProtKB for these proteins.")

    with st.expander("About the data"):
        st.markdown(
            "Disease associations come from the curated *Involvement in disease* section of "
            "UniProtKB/Swiss-Prot entries (with OMIM cross-references). Unreviewed "
            "(TrEMBL) entries usually have none. Disease-area filters are simple keyword "
            "matches on the disease names.")
