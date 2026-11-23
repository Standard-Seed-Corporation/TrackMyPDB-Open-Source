"""Offline tests for input parsing, structure parsing and the extraction pipeline."""

import pytest

from trackmypdb.extraction import (EXCLUDED_CODES, RECORD_COLUMNS, HeteroatomExtractor,
                                   is_valid_uniprot_accession, parse_mmcif_hetatm,
                                   parse_pdb_hetatm, parse_uniprot_ids)


# ------------------------------------------------------------------ inputs --
@pytest.mark.parametrize("acc", ["P37231", "Q9UNQ0", "A0A023GPI8", "P04637-2", "p37231"])
def test_valid_accessions(acc):
    assert is_valid_uniprot_accession(acc)


@pytest.mark.parametrize("acc", ["", "BAD", "12345", "P3723", "1ABC"])
def test_invalid_accessions(acc):
    assert not is_valid_uniprot_accession(acc)


def test_parse_uniprot_ids_mixed_separators_and_duplicates():
    assert parse_uniprot_ids("p37231, Q9UNQ0\nP37231;  P06276 ") == ["P37231", "Q9UNQ0", "P06276"]
    assert parse_uniprot_ids(["P37231", " ", "q9unq0"]) == ["P37231", "Q9UNQ0"]


# ----------------------------------------------------------------- parsing --
def test_parse_mmcif(mmcif_1abc):
    groups = parse_mmcif_hetatm(mmcif_1abc)
    assert set(groups) == {"MSE", "BRL", "SO4", "HOH"}
    brl = groups["BRL"]
    assert brl.chains == {"A", "B"}
    assert brl.residues == {("A", "501"), ("B", "502")}
    assert "O5'" in brl.atom_names  # quoted atom name keeps its prime


def test_parse_mmcif_ignores_other_loops():
    text = "data_X\nloop_\n_other.a\n_other.b\nHETATM X\n#\n"
    assert parse_mmcif_hetatm(text) == {}


def test_parse_pdb(pdb_2xyz):
    groups = parse_pdb_hetatm(pdb_2xyz.splitlines())
    assert set(groups) == {"BRL", "GSH", "HOH"}
    assert groups["GSH"].atom_names == {"N1", "CA"}
    assert groups["GSH"].residues == {("A", "601")}


def test_water_and_ions_are_excluded_by_default():
    assert {"HOH", "SO4", "NA", "CL", "GOL"} <= EXCLUDED_CODES


# --------------------------------------------------------------- pipeline --
def make_extractor(fake_api, **kw):
    return HeteroatomExtractor(session_factory=lambda: fake_api, workers=2, **kw)


def test_extract_end_to_end(fake_api):
    seen = []
    result = make_extractor(fake_api).extract("P37231", progress=lambda f, m: seen.append(f))
    df = result.records

    assert list(df.columns) == RECORD_COLUMNS
    # duplicates in the PDBe answer are removed, PDBe order kept
    assert result.pdbs_per_uniprot == {"P37231": ["1ABC", "2XYZ"]}
    assert result.failed_structures == []
    # water, sulfate and the polymer component MSE are dropped
    assert set(df["Heteroatom_Code"]) == {"BRL", "GSH"}
    brl_1abc = df[(df.PDB_ID == "1ABC") & (df.Heteroatom_Code == "BRL")].iloc[0]
    assert brl_1abc.Chemical_Name == "ROSIGLITAZONE"
    assert brl_1abc.Source == "rcsb"
    assert brl_1abc.Copies == 2
    assert brl_1abc.InChIKey.startswith("YASAKCUCGLMORW")
    # GSH is only in the legacy-format file and its online lookup fails
    gsh = df[df.Heteroatom_Code == "GSH"].iloc[0]
    assert gsh.PDB_ID == "2XYZ" and gsh.Source == "local_db" and gsh.SMILES
    # progress is monotonic and finishes at 1.0
    assert seen == sorted(seen) and seen[-1] == 1.0


def test_each_ligand_is_looked_up_once(fake_api):
    make_extractor(fake_api).extract(["P37231"])
    chemcomp_calls = [u for u in fake_api.calls if "/chemcomp/BRL" in u]
    assert len(chemcomp_calls) == 1  # BRL occurs in both structures


def test_keep_everything_option(fake_api):
    df = make_extractor(fake_api, exclude_codes=frozenset(),
                        exclude_polymer_components=False).extract("P37231").records
    assert {"HOH", "SO4", "MSE", "BRL", "GSH"} <= set(df["Heteroatom_Code"])


def test_max_structures_uses_best_ranked_first(fake_api):
    result = make_extractor(fake_api).extract("P37231", max_structures=1)
    assert result.pdbs_per_uniprot["P37231"] == ["1ABC"]


def test_no_structures_returns_empty_frame_not_error(fake_api):
    result = make_extractor(fake_api).extract(["Q9UNQ0", "NOT_AN_ID"])
    assert result.records.empty
    assert list(result.records.columns) == RECORD_COLUMNS
    assert result.invalid_ids == ["NOT_AN_ID"]
    assert result.unmapped_ids == ["Q9UNQ0"]
    assert result.summary["records"] == 0


def test_failed_download_is_reported(fake_api):
    from tests.conftest import FakeResponse
    from trackmypdb import extraction as ex
    fake_api.routes[ex.RCSB_PDB_URL.format(pdb="2XYZ")] = FakeResponse(500)
    result = make_extractor(fake_api).extract("P37231")
    assert result.failed_structures == ["2XYZ"]
    assert set(result.records["PDB_ID"]) == {"1ABC"}


def test_local_db_fallback_can_be_disabled(fake_api):
    df = make_extractor(fake_api, use_local_db=False).extract("P37231").records
    gsh = df[df.Heteroatom_Code == "GSH"].iloc[0]
    assert gsh.SMILES == "" and gsh.Source == "not_found"
