"""Tests for the bundled ligand database and protein/disease annotations."""

import pandas as pd

from tests.conftest import FakeResponse, FakeSession
from trackmypdb import annotations as an
from trackmypdb.ligand_db import LIGAND_DB_COLUMNS, ligand_lookup, load_ligand_db


def test_bundled_database_loads():
    db = load_ligand_db()
    assert list(db.columns)[:6] == LIGAND_DB_COLUMNS
    assert len(db) > 3000
    assert db["PDB_ID"].nunique() > 2000
    with_smiles = load_ligand_db(with_smiles_only=True)
    assert (with_smiles["SMILES"] != "").all()
    assert len(with_smiles) < len(db)


def test_ligand_lookup():
    hit = ligand_lookup()["GSH"]
    assert hit["Chemical_Name"] == "GLUTATHIONE"
    assert hit["SMILES"]


def test_fetch_pdb_protein_info_batches_and_parses():
    payload = {"data": {"entries": [
        {"rcsb_id": "1ABC", "polymer_entities": [{
            "rcsb_polymer_entity": {"pdbx_description": "Peroxisome proliferator-activated receptor gamma"},
            "rcsb_polymer_entity_container_identifiers": {
                "uniprot_ids": ["P37231"],
                "reference_sequence_identifiers": [
                    {"database_name": "UniProt", "database_accession": "P37231"},
                    {"database_name": "UniProt", "database_accession": "Q15596"}]}}]},
        None]}}
    session = FakeSession({an.RCSB_GRAPHQL_URL: FakeResponse(json_data=payload)})
    info = an.fetch_pdb_protein_info(["1abc", "9ZZZ", "1ABC"], batch_size=1, session=session)
    assert len(session.calls) == 2  # two unique IDs, batch size 1
    assert info["1ABC"]["uniprot_ids"] == ["P37231", "Q15596"]
    assert info["9ZZZ"] == {"uniprot_ids": [], "protein_names": []}


def test_parse_uniprot_entry():
    entry = {
        "proteinDescription": {"recommendedName": {"fullName": {"value": "Cellular tumor antigen p53"}}},
        "genes": [{"geneName": {"value": "TP53"}}],
        "comments": [
            {"commentType": "FUNCTION"},
            {"commentType": "DISEASE", "disease": {
                "diseaseId": "Li-Fraumeni syndrome", "acronym": "LFS",
                "description": "Autosomal dominant familial cancer syndrome",
                "diseaseCrossReference": {"database": "MIM", "id": "151623"}}}]}
    parsed = an.parse_uniprot_entry(entry)
    assert parsed["protein_name"] == "Cellular tumor antigen p53"
    assert parsed["genes"] == ["TP53"]
    assert parsed["diseases"][0]["name"] == "Li-Fraumeni syndrome"
    assert parsed["diseases"][0]["mim"] == "151623"


def test_add_disease_annotations(monkeypatch):
    fake = {"P04637": {"diseases": [{"name": "Li-Fraumeni syndrome"}]},
            "P37231": {"diseases": []}}
    monkeypatch.setattr(an, "fetch_uniprot_diseases", lambda accs: fake)
    df = pd.DataFrame({"UniProt_IDs": ["P04637", "P37231", "N/A", "P04637, P37231"]})
    out = an.add_disease_annotations(df)
    assert out["Disease_Associations"].tolist() == [
        "Li-Fraumeni syndrome", "None reported", "None reported", "Li-Fraumeni syndrome"]


def test_keyword_matching():
    assert an.matches_keywords("Li-Fraumeni syndrome; breast CANCER", an.DISEASE_CATEGORIES["Cancer"])
    assert not an.matches_keywords("None reported", ["cancer"])
    assert not an.matches_keywords("anything", ["", "  "])
