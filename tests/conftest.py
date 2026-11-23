"""Shared fixtures: an in-memory fake of the RCSB/PDBe/UniProt HTTP APIs so the
test-suite runs offline and deterministically."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA = Path(__file__).parent / "data"


def pytest_addoption(parser):
    parser.addoption("--run-network", action="store_true",
                     help="also run tests that call the live public APIs")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="needs --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


class FakeResponse:
    def __init__(self, status_code=200, text="", json_data=None):
        self.status_code = status_code
        self._json = json_data
        self.text = text if json_data is None else json.dumps(json_data)

    def json(self):
        if self._json is None:
            return json.loads(self.text)
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Maps URL -> FakeResponse (or an exception to raise); records calls."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls: list[str] = []

    def _lookup(self, url):
        self.calls.append(url)
        route = self.routes.get(url, FakeResponse(404))
        if isinstance(route, Exception):
            raise route
        return route

    def get(self, url, timeout=None, **kw):
        return self._lookup(url)

    def post(self, url, json=None, timeout=None, **kw):
        return self._lookup(url)


@pytest.fixture
def mmcif_1abc() -> str:
    return (DATA / "1ABC_minimal.cif").read_text()


@pytest.fixture
def pdb_2xyz() -> str:
    return (DATA / "2XYZ_minimal.pdb").read_text()


@pytest.fixture
def fake_api(mmcif_1abc, pdb_2xyz):
    """A fake API with one UniProt -> two PDB entries (one mmCIF, one only in
    legacy PDB format) containing rosiglitazone (BRL), water, sulfate,
    selenomethionine and an unknown ligand only present in the local DB."""
    from trackmypdb import extraction as ex

    routes = {
        ex.PDBE_BEST_STRUCTURES_URL.format(uniprot="P37231"): FakeResponse(json_data={
            "P37231": [{"pdb_id": "1abc"}, {"pdb_id": "2xyz"}, {"pdb_id": "1abc"}]}),
        ex.RCSB_MMCIF_URL.format(pdb="1ABC"): FakeResponse(text=mmcif_1abc),
        ex.RCSB_MMCIF_URL.format(pdb="2XYZ"): FakeResponse(503),
        ex.RCSB_PDB_URL.format(pdb="2XYZ"): FakeResponse(text=pdb_2xyz),
        ex.RCSB_CHEMCOMP_URL.format(code="BRL"): FakeResponse(json_data={
            "chem_comp": {"name": "ROSIGLITAZONE", "formula": "C18 H19 N3 O3 S",
                          "type": "NON-POLYMER"},
            "rcsb_chem_comp_descriptor": {
                "smiles": "CN(CCOc1ccc(CC2SC(=O)NC2=O)cc1)c1ccccn1",
                "in_ch_ikey": "YASAKCUCGLMORW-UHFFFAOYSA-N"}}),
        ex.RCSB_CHEMCOMP_URL.format(code="MSE"): FakeResponse(json_data={
            "chem_comp": {"name": "SELENOMETHIONINE", "formula": "C5 H11 N O2 Se",
                          "type": "L-PEPTIDE LINKING"},
            "rcsb_chem_comp_descriptor": {"smiles": "C[Se]CC[C@H](N)C(=O)O"}}),
        # GSH: online lookup fails -> must come from the bundled database
        ex.RCSB_CHEMCOMP_URL.format(code="GSH"): requests.ConnectionError("offline"),
    }
    return FakeSession(routes)
