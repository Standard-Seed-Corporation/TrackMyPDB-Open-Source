"""Similarity and descriptor tests (need RDKit)."""

import math

import pandas as pd
import pytest

pytest.importorskip("rdkit")

from trackmypdb.properties import calculate_properties, draw_molecule  # noqa: E402
from trackmypdb.similarity import (InvalidSmilesError, compare_smiles, fingerprint,  # noqa: E402
                                   fingerprints, rank_by_similarity)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
SALICYLIC_ACID = "Oc1ccccc1C(=O)O"
BENZENE = "c1ccccc1"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"


@pytest.fixture
def library():
    return pd.DataFrame({
        "PDB_ID": ["1AAA", "1BBB", "1CCC", "1DDD", "1EEE"],
        "Heteroatom_Code": ["ASA", "SAL", "BNZ", "CFF", "BAD"],
        "SMILES": [ASPIRIN, SALICYLIC_ACID, BENZENE, CAFFEINE, "not a smiles"],
    })


def test_identical_molecules_score_one():
    assert compare_smiles("CCO", "OCC") == pytest.approx(1.0)


def test_similarity_is_symmetric_and_bounded():
    a, b = compare_smiles(ASPIRIN, CAFFEINE), compare_smiles(CAFFEINE, ASPIRIN)
    assert a == pytest.approx(b)
    assert 0.0 <= a < 1.0


def test_invalid_smiles_raises():
    with pytest.raises(InvalidSmilesError):
        compare_smiles("not_a_molecule", "CCO")
    assert fingerprint("") is None and fingerprint(None) is None


def test_fingerprint_settings_change_bits():
    assert fingerprint(ASPIRIN, n_bits=1024).GetNumBits() == 1024
    assert fingerprint(ASPIRIN, radius=1).GetNumBits() == 2048


def test_ranking_order_and_columns(library):
    hits = rank_by_similarity(ASPIRIN, library, top_n=None)
    assert hits.iloc[0]["Heteroatom_Code"] == "ASA"
    assert hits.iloc[0]["Tanimoto_Similarity"] == pytest.approx(1.0)
    assert hits.iloc[1]["Heteroatom_Code"] == "SAL"
    assert hits["Tanimoto_Similarity"].is_monotonic_decreasing
    assert "BAD" not in set(hits["Heteroatom_Code"])          # unparseable row dropped
    assert {"PDB_ID", "Heteroatom_Code", "SMILES"} <= set(hits.columns)  # input kept


def test_top_n_is_respected_beyond_ten():
    """Regression: v2.0 silently capped results at 10 rows."""
    big = pd.DataFrame({"PDB_ID": [f"{i}X" for i in range(40)],
                        "Heteroatom_Code": ["L"] * 40,
                        "SMILES": ["C" * (i % 10 + 1) + "O" for i in range(40)]})
    assert len(rank_by_similarity("CCCO", big, top_n=25)) == 25
    assert len(rank_by_similarity("CCCO", big, top_n=None)) == 40


def test_min_similarity_filter(library):
    hits = rank_by_similarity(ASPIRIN, library, top_n=None, min_similarity=0.99)
    assert hits["Heteroatom_Code"].tolist() == ["ASA"]


def test_precomputed_fingerprints_give_same_result(library):
    fps = fingerprints(library["SMILES"])
    a = rank_by_similarity(SALICYLIC_ACID, library, top_n=None)
    b = rank_by_similarity(SALICYLIC_ACID, library, top_n=None, precomputed_fps=fps)
    pd.testing.assert_frame_equal(a, b)


def test_invalid_target_raises(library):
    with pytest.raises(InvalidSmilesError):
        rank_by_similarity("xyz(", library)


def test_empty_library():
    hits = rank_by_similarity(ASPIRIN, pd.DataFrame(columns=["PDB_ID", "SMILES"]))
    assert hits.empty and "Tanimoto_Similarity" in hits.columns


def test_properties_aspirin():
    p = calculate_properties(ASPIRIN)
    assert math.isclose(p["Molecular_Weight"], 180.16, abs_tol=0.01)
    assert p["HBD"] == 1 and p["Lipinski_Violations"] == 0
    assert p["Formula"] == "C9H8O4"


def test_draw_molecule():
    assert draw_molecule(ASPIRIN, size=(200, 200)).size == (200, 200)
    with pytest.raises(InvalidSmilesError):
        draw_molecule("nope")


# ------------------------------------------------------------ substructure --
from trackmypdb.substructure import (match_atoms, maximum_common_substructure,  # noqa: E402
                                     parse_query, substructure_search)


def test_substructure_search_smiles_query(library):
    hits = substructure_search("c1ccccc1", library)
    # benzene ring: aspirin, salicylic acid, benzene, NOT caffeine (no benzene ring)
    assert set(hits["Heteroatom_Code"]) == {"ASA", "SAL", "BNZ"}
    assert hits.iloc[0]["Heteroatom_Code"] == "BNZ"            # most similar first
    assert hits["Tanimoto_Similarity"].is_monotonic_decreasing


def test_substructure_search_smarts_query(library):
    hits = substructure_search("[CX3](=O)[OX2H1]", library)       # carboxylic acid
    assert set(hits["Heteroatom_Code"]) == {"ASA", "SAL"}
    assert "Tanimoto_Similarity" not in hits.columns            # SMARTS only
    assert hits["Heavy_Atoms"].is_monotonic_increasing          # smallest first


def test_substructure_search_limits_and_errors(library):
    assert len(substructure_search("c1ccccc1", library, max_results=1)) == 1
    with pytest.raises(InvalidSmilesError):
        substructure_search("[invalid", library)
    assert parse_query("") is None


def test_maximum_common_substructure_and_highlight():
    smarts = maximum_common_substructure(ASPIRIN, SALICYLIC_ACID)
    assert smarts
    atoms, bonds = match_atoms(SALICYLIC_ACID, smarts)
    assert len(atoms) == 10            # all heavy atoms of salicylic acid are shared
    assert bonds
    assert maximum_common_substructure("CCO", "nonsense(") is None
