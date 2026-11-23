"""2D structure drawings (SVG) and visual hit cards."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

EXAMPLES = {
    "Imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "Gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "Sildenafil": "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12",
    "Methotrexate": "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1",
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Estradiol": "CC12CCC3c4ccc(O)cc4CCC3C1CCC2O",
}


@st.cache_data(max_entries=2000, show_spinner=False)
def mol_svg(smiles: str, width: int = 260, height: int = 190, white: bool = False,
            highlight: str | None = None) -> str:
    """Inline, responsive SVG depiction of a molecule ("" if it cannot be drawn).

    ``white=True`` draws white lines on a transparent background (for dark banners).
    ``highlight`` is a SMARTS pattern whose matched atoms/bonds are shaded green.
    """
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    from trackmypdb.similarity import mol_from_smiles

    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    try:
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = False
        opts.bondLineWidth = 2
        opts.padding = 0.08
        if white:
            opts.useBWAtomPalette()
        atoms, bonds = [], []
        if highlight:
            from trackmypdb.substructure import match_atoms
            atoms, bonds = match_atoms(smiles, highlight)
        if atoms:
            green = (0.66, 0.87, 0.67)
            rdMolDraw2D.PrepareAndDrawMolecule(
                drawer, mol, highlightAtoms=atoms, highlightBonds=bonds,
                highlightAtomColors={a: green for a in atoms},
                highlightBondColors={b: green for b in bonds})
        else:
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
    except Exception:  # noqa: BLE001 - drawing is cosmetic
        return ""
    svg = svg[svg.find("<svg"):]                       # drop the XML declaration
    svg = svg.replace(f"width='{width}px' height='{height}px'", "width='100%'")
    if white:
        svg = svg.replace("#000000", "#FFFFFF")
    return " ".join(svg.split())                        # one line, safe for markdown


@st.cache_data(max_entries=2000, show_spinner=False)
def shared_fragment(query: str, smiles: str) -> str | None:
    """SMARTS of the part a hit shares with the query (cached)."""
    from trackmypdb.substructure import maximum_common_substructure
    return maximum_common_substructure(query, smiles, timeout=1)


def _score_class(score: float) -> str:
    return "tmp-high" if score >= 0.7 else "tmp-mid" if score >= 0.4 else "tmp-low"


def hit_cards(hits: pd.DataFrame, n: int = 8, query_smiles: str | None = None,
              show_query: bool = True, highlight: str = "shared") -> None:
    """Grid of structure cards for the best hits.

    highlight: ``"shared"`` shades the atoms each hit shares with the query
    (maximum common substructure); any other string is used as a SMARTS
    pattern (e.g. the substructure query); ``""`` disables highlighting.
    """
    cards = []
    if query_smiles and show_query:
        cards.append(
            '<div class="tmp-card tmp-query">'
            f'<div class="tmp-mol">{mol_svg(query_smiles)}</div>'
            '<div class="tmp-card-title">Your molecule</div>'
            '<div class="tmp-card-name">query</div></div>')
    for row in hits.head(n).itertuples(index=False):
        pdb = html.escape(str(getattr(row, "PDB_ID", "")))
        code = html.escape(str(getattr(row, "Heteroatom_Code", "")))
        name = html.escape(str(getattr(row, "Chemical_Name", "") or "").title())
        score = getattr(row, "Tanimoto_Similarity", None)
        q = getattr(row, "Query_SMILES", None) or query_smiles
        if highlight == "shared":
            patt = shared_fragment(q, str(row.SMILES)) if q else None
        else:
            patt = highlight or None
        links = (f'<a href="https://www.rcsb.org/structure/{pdb}" target="_blank">{pdb}</a>'
                 f' · <a href="https://www.rcsb.org/ligand/{code}" target="_blank">{code}</a>')
        cards.append(
            '<div class="tmp-card">'
            + (f'<span class="tmp-score {_score_class(float(score))}">{float(score):.2f}</span>'
               if score is not None and score == score else
               '<span class="tmp-score tmp-high">match</span>')
            + f'<div class="tmp-mol">{mol_svg(str(row.SMILES), highlight=patt)}</div>'
            f'<div class="tmp-card-title">{links}</div>'
            f'<div class="tmp-card-name" title="{name}">{name}</div></div>')
    st.markdown('<div class="tmp-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


@st.cache_data(max_entries=500, show_spinner=False)
def _props(smiles: str) -> dict | None:
    from trackmypdb.properties import calculate_properties
    try:
        return calculate_properties(smiles)
    except Exception:  # noqa: BLE001
        return None


def query_preview(smiles: str | None) -> None:
    """Small live card: structure drawing plus key descriptors of the query."""
    if not smiles:
        st.markdown('<div class="tmp-preview"><div class="tmp-mol" style="color:#9E9E9E">'
                    "Your molecule will appear here</div></div>", unsafe_allow_html=True)
        return
    svg = mol_svg(smiles, 300, 200)
    if not svg:  # e.g. a SMARTS pattern: show it as text
        st.markdown('<div class="tmp-preview"><div class="tmp-mol" style="font-family:monospace;'
                    f'color:#1B5E20">{html.escape(smiles)}</div></div>', unsafe_allow_html=True)
        return
    p = _props(smiles)
    chips = ""
    if p:
        lip = p["Lipinski_Violations"]
        chips = "".join(f'<span class="tmp-prop">{k} {v}</span>' for k, v in [
            ("MW", p["Molecular_Weight"]), ("cLogP", p["LogP"]), ("TPSA", p["TPSA"])])
        chips += (f'<span class="tmp-prop{" warn" if lip else ""}">'
                  f'{"Ro5 ✓" if not lip else f"Ro5 ✗{lip}"}</span>')
    st.markdown(f'<div class="tmp-preview"><div class="tmp-mol">{svg}</div>'
                f'<div class="tmp-props">{chips}</div></div>', unsafe_allow_html=True)
