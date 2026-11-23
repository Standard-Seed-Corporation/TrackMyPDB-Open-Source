"""Integration tests against the live public APIs.

Skipped by default; run with ``pytest --run-network``. Useful before a release
to detect upstream API changes (RCSB, PDBe, UniProt).
"""

import pytest

from trackmypdb import HeteroatomExtractor
from trackmypdb.annotations import fetch_pdb_protein_info, fetch_uniprot_diseases

pytestmark = pytest.mark.network


def test_pdbe_mapping_returns_structures():
    pdbs = HeteroatomExtractor().get_pdbs_for_uniprot("P37231")  # PPAR-gamma
    assert "2PRG" in pdbs


def test_rcsb_ligand_lookup():
    info = HeteroatomExtractor(use_local_db=False).lookup_ligand("BRL")
    assert info.source in {"rcsb", "rcsb_sdf"} and info.smiles
    assert "ROSIGLITAZONE" in info.name.upper()


def test_extraction_finds_rosiglitazone_in_2prg():
    ex = HeteroatomExtractor()
    hets = ex.fetch_structure_hets("2PRG")
    assert "BRL" in hets


def test_protein_and_disease_annotation():
    info = fetch_pdb_protein_info(["2PRG"])
    assert "P37231" in info["2PRG"]["uniprot_ids"]
    entry = fetch_uniprot_diseases(["P37231"])["P37231"]
    assert entry["status"] == "ok" and entry["diseases"]
