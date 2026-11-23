"""Tests for the command-line interface and the MCP tool functions."""

import pandas as pd
import pytest

from trackmypdb import cli


def test_parser_subcommands():
    p = cli.build_parser()
    args = p.parse_args(["extract", "P37231", "Q9UNQ0", "--max-structures", "3"])
    assert args.uniprot_ids == ["P37231", "Q9UNQ0"] and args.max_structures == 3
    args = p.parse_args(["search", "CCO", "--top", "5", "--min-similarity", "0.4"])
    assert args.top == 5 and args.min_similarity == 0.4


def test_cli_extract_writes_csv(tmp_path, monkeypatch, fake_api):
    from trackmypdb import extraction
    monkeypatch.setattr(extraction, "get_session", lambda: fake_api)
    out = tmp_path / "ligands.csv"
    assert cli.main(["extract", "P37231", "-o", str(out)]) == 0
    df = pd.read_csv(out)
    assert set(df["Heteroatom_Code"]) == {"BRL", "GSH"}


def test_cli_search_and_compare(tmp_path, capsys):
    pytest.importorskip("rdkit")
    out = tmp_path / "hits.csv"
    # glutathione is in the bundled database, so it must be found with similarity 1
    assert cli.main(["search", "C(CC(=O)NC(CS)C(=O)NCC(=O)O)C(C(=O)O)N", "--top", "5",
                     "-o", str(out)]) == 0
    hits = pd.read_csv(out)
    assert hits.iloc[0]["Heteroatom_Code"] == "GSH"
    assert hits.iloc[0]["Tanimoto_Similarity"] == pytest.approx(1.0)

    sub_out = tmp_path / "sub.csv"
    assert cli.main(["substructure", "c1ccc2ncncc2c1", "--top", "5", "-o", str(sub_out)]) == 0
    assert "Heavy_Atoms" in pd.read_csv(sub_out).columns
    assert cli.main(["substructure", "[not valid"]) == 2

    assert cli.main(["compare", "CCO", "CCO"]) == 0
    assert capsys.readouterr().out.strip() == "1.0000"
    assert cli.main(["compare", "CCO", "nonsense("]) == 2


def test_mcp_core_tools():
    pytest.importorskip("rdkit")
    from mcp_server import core

    assert core.compare_smiles("CCO", "CCO")["tanimoto_similarity"] == 1.0
    bad = core.compare_smiles("not_a_molecule", "CCO")
    assert "error" in bad and bad["valid_a"] is False and bad["valid_b"] is True

    ranked = core.analyze_similarity("CCO", [{"PDB_ID": "1A", "Heteroatom_Code": "EOH",
                                              "SMILES": "CCO"}])
    assert ranked["result_count"] == 1 and ranked["results"][0]["Tanimoto_Similarity"] == 1.0
    assert "error" in core.analyze_similarity("CCO", [])

    db = core.search_ligand_database("C(CC(=O)NC(CS)C(=O)NCC(=O)O)C(C(=O)O)N", top_n=3)
    assert db["results"][0]["Heteroatom_Code"] == "GSH"
