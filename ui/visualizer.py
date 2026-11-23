"""Molecule viewer page: 2D depiction and descriptors."""

import re
from io import BytesIO

import pandas as pd
import streamlit as st

from trackmypdb.properties import calculate_properties, draw_molecule
from ui.molecules import mol_svg
from ui.style import page_header
from ui.common import smiles_feedback, smiles_lines, timestamp


def _from_upload(upload) -> str | None:
    from rdkit import Chem
    content = upload.getvalue().decode("utf-8", errors="replace")
    if upload.name.lower().endswith((".mol", ".sdf")):
        mol = Chem.MolFromMolBlock(content)
        return Chem.MolToSmiles(mol) if mol is not None else None
    lines = smiles_lines(content.replace(",", " "))
    return lines[0] if lines else None


def render() -> None:
    page_header("Molecule Viewer", "Structure and drug-likeness at a glance.", icon="🧪")
    tab1, tab2 = st.tabs(["Enter SMILES", "Upload file"])
    with tab1:
        typed = st.text_input("SMILES", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")
    with tab2:
        upload = st.file_uploader("MOL/SDF, or a text/CSV file whose first entry is a SMILES",
                                  type=["mol", "sdf", "smi", "txt", "csv"])
    smiles = _from_upload(upload) if upload else (smiles_lines(typed) or [None])[0]
    if not smiles:
        return
    if not smiles_feedback([smiles]):
        return

    props = calculate_properties(smiles)
    left, right = st.columns([1, 1])
    with left:
        st.markdown(f'<div class="tmp-card"><div class="tmp-mol" style="height:340px">'
                    f'{mol_svg(smiles, 460, 340)}</div></div>', unsafe_allow_html=True)
        buf = BytesIO()
        draw_molecule(smiles, size=(1000, 1000)).save(buf, format="PNG")
        safe = re.sub(r"[^A-Za-z0-9]+", "_", smiles)[:30].strip("_") or "molecule"
        st.download_button("Download image (PNG)", buf.getvalue(), f"{safe}_{timestamp()}.png",
                           "image/png", on_click="ignore")
    with right:
        st.code(smiles, language="text")
        c = st.columns(3)
        c[0].metric("MW (Da)", props["Molecular_Weight"])
        c[1].metric("cLogP", props["LogP"])
        c[2].metric("TPSA (Å²)", props["TPSA"])
        c = st.columns(3)
        c[0].metric("H-bond donors", props["HBD"])
        c[1].metric("H-bond acceptors", props["HBA"])
        c[2].metric("Rotatable bonds", props["Rotatable_Bonds"])
        st.caption(f"Formula {props['Formula']} · {props['Heavy_Atoms']} heavy atoms")
        v = props["Lipinski_Violations"]
        msg = f"{v} Lipinski rule-of-five violation(s)"
        (st.success if v == 0 else st.warning if v == 1 else st.error)(msg)
        table = pd.DataFrame([{"SMILES": smiles, **props}])
        st.download_button("Download properties (CSV)", table.to_csv(index=False).encode(),
                           f"properties_{safe}_{timestamp()}.csv", "text/csv", on_click="ignore")
