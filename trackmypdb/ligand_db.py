"""The bundled offline ligand database (``trackmypdb/data/pdb_ligands.csv``).

It maps PDB entries to their co-crystallised ligands with SMILES, names and
formulae. It is used for the "Quick Search" page and as an offline
fallback when the online chemical-component services cannot be reached.
"""

from __future__ import annotations

import functools
from importlib import resources
from pathlib import Path

import pandas as pd

LIGAND_DB_COLUMNS = ["PDB_ID", "Heteroatom_Code", "SMILES", "Chemical_Name",
                     "Formula", "Component_Type"]


def default_db_path() -> Path:
    return Path(str(resources.files("trackmypdb") / "data" / "pdb_ligands.csv"))


def load_ligand_db(path: str | Path | None = None, with_smiles_only: bool = False) -> pd.DataFrame:
    """Load the ligand database as a DataFrame.

    The historical CSV stores the chemical-component *type* (e.g.
    ``non-polymer``) in a column called ``Status``; it is renamed to
    ``Component_Type`` here so it is not confused with the extractor's
    lookup status.
    """
    df = pd.read_csv(path or default_db_path(), dtype=str, keep_default_na=False)
    if "Status" in df.columns and "Component_Type" not in df.columns:
        df = df.rename(columns={"Status": "Component_Type"})
    for col in LIGAND_DB_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df["SMILES"] = df["SMILES"].str.strip()
    if with_smiles_only:
        df = df[df["SMILES"] != ""].reset_index(drop=True)
    return df


@functools.lru_cache(maxsize=4)
def ligand_lookup(path: str | None = None) -> dict:
    """Map ``Heteroatom_Code`` to ``{SMILES, Chemical_Name, Formula,
    Component_Type}`` using the first database row that has a SMILES."""
    df = load_ligand_db(path, with_smiles_only=True)
    df = df.drop_duplicates("Heteroatom_Code", keep="first")
    return {
        row.Heteroatom_Code: {
            "SMILES": row.SMILES,
            "Chemical_Name": row.Chemical_Name,
            "Formula": row.Formula,
            "Component_Type": row.Component_Type,
        }
        for row in df.itertuples(index=False)
    }
