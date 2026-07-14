"""
Offline sanity tests for the TrackMyPDB core tools.

Only compare_smiles is tested here because it needs no network. Run with:

    pip install pytest
    python -m pytest tests/test_core.py -v
"""

from mcp_server import core


def test_identical_molecules_are_perfectly_similar():
    res = core.compare_smiles("CCO", "CCO")  # ethanol vs ethanol
    assert res["tanimoto_similarity"] == 1.0


def test_different_molecules_are_less_similar():
    res = core.compare_smiles("CCO", "c1ccccc1")  # ethanol vs benzene
    assert res["tanimoto_similarity"] < 1.0


def test_invalid_smiles_reported():
    res = core.compare_smiles("not_a_molecule", "CCO")
    assert "error" in res
    assert res["valid_b"] is True
