"""Interactive 3D view of a ligand in its binding site (3Dmol.js, runs in the browser)."""

from __future__ import annotations

import json
import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

_SAFE_ID = re.compile(r"^[A-Za-z0-9]{1,12}$")

_TEMPLATE = """
<div id="v" style="width:100%;height:__H__px;position:relative;border-radius:14px;
     overflow:hidden;border:1px solid #E3EDE4;background:#fff"></div>
<div id="msg" style="font:13px sans-serif;color:#5F6368;margin-top:4px"></div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
if (typeof $3Dmol === "undefined") {
  document.getElementById("msg").textContent =
    "The 3D viewer could not be loaded. It needs an internet connection.";
  throw new Error("3Dmol.js not available");
}
const pdb = __PDB__, lig = __LIG__;
const msg = document.getElementById("msg");
msg.textContent = "Loading " + pdb + " from RCSB PDB...";
const viewer = $3Dmol.createViewer("v", {backgroundColor: "white"});
fetch("https://files.rcsb.org/download/" + pdb + ".cif")
  .then(r => { if (!r.ok) throw new Error(r.status); return r.text(); })
  .then(data => {
    viewer.addModel(data, "cif");
    const ligand = {resn: lig};
    viewer.setStyle({}, {cartoon: {color: "spectrum", opacity: 0.8}});
    viewer.setStyle({within: {distance: 5, sel: ligand}, byres: true, not: {resn: "HOH"}},
                    {cartoon: {color: "spectrum", opacity: 0.8},
                     stick: {radius: 0.14, colorscheme: "grayCarbon"}});
    viewer.setStyle(ligand, {stick: {radius: 0.28, colorscheme: "greenCarbon"}});
    viewer.addSurface($3Dmol.SurfaceType.VDW, {opacity: 0.12, color: "#2E7D32"}, ligand);
    viewer.zoomTo(ligand);
    viewer.render();
    msg.textContent = pdb + " · " + lig + " (green) with residues within 5 Å. "
                    + "Drag to rotate, scroll to zoom.";
  })
  .catch(e => { msg.textContent = "Could not load " + pdb + " (" + e + "). "
                                 + "The 3D view needs an internet connection."; });
</script>
"""


def binding_site_view(pdb_id: str, ligand_code: str, height: int = 460) -> None:
    """Render the 3D viewer for one PDB entry and ligand code."""
    if not (_SAFE_ID.match(str(pdb_id)) and _SAFE_ID.match(str(ligand_code))):
        st.info("No 3D view available for this entry.")
        return
    page = (_TEMPLATE.replace("__H__", str(height))
            .replace("__PDB__", json.dumps(str(pdb_id).upper()))
            .replace("__LIG__", json.dumps(str(ligand_code).upper())))
    components.html(page, height=height + 30)


def structure_viewer(hits: pd.DataFrame, key: str, n: int = 25) -> None:
    """Section with a hit picker and the 3D binding-site view."""
    rows = hits.head(n)
    if rows.empty or not {"PDB_ID", "Heteroatom_Code"} <= set(rows.columns):
        return
    st.subheader(":material/view_in_ar: 3D binding site")
    labels = []
    for r in rows.itertuples(index=False):
        sim = getattr(r, "Tanimoto_Similarity", None)
        score = f" · {float(sim):.2f}" if sim is not None and sim == sim else ""
        name = str(getattr(r, "Chemical_Name", "") or "")[:40].title()
        labels.append(f"{r.PDB_ID} · {r.Heteroatom_Code}{score}  {name}")
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    idx = c1.selectbox("Hit to view", range(len(labels)), format_func=lambda i: labels[i],
                       key=f"{key}_3d")
    row = rows.iloc[idx]
    c2.link_button("Open in RCSB", f"https://www.rcsb.org/3d-view/{row['PDB_ID']}",
                   icon=":material/open_in_new:")
    binding_site_view(row["PDB_ID"], row["Heteroatom_Code"])
