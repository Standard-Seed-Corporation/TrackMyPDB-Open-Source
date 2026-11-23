"""
Network diagnostic for TrackMyPDB's SMILES sources.

Checks whether the external chemical-data endpoints are reachable from THIS
machine/network, so you can tell "flaky/slow" apart from "blocked by firewall".

Run:
    python scripts/net_diag.py
"""

import time

import requests

CODE = "NAG"  # a very common ligand that definitely exists in every source

ENDPOINTS = [
    ("PDBe best_structures", "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/P37231"),
    ("RCSB mmCIF download", "https://files.rcsb.org/download/1PRG.cif"),
    ("RCSB chemcomp (SMILES)", f"https://data.rcsb.org/rest/v1/core/chemcomp/{CODE}"),
    ("RCSB ligand SDF", f"https://files.rcsb.org/ligands/download/{CODE}_ideal.sdf"),
    ("RCSB GraphQL", "https://data.rcsb.org/graphql?query=%7Bentry(entry_id:%221PRG%22)%7Brcsb_id%7D%7D"),
    ("UniProt REST", "https://rest.uniprot.org/uniprotkb/P37231.json"),
]


def main():
    print(f"Testing {len(ENDPOINTS)} endpoints (timeout 20s each)...\n")
    for name, url in ENDPOINTS:
        start = time.time()
        try:
            r = requests.get(url, timeout=20)
            dt = time.time() - start
            print(f"  [{r.status_code}] {name}  ({dt:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            dt = time.time() - start
            print(f"  [FAIL] {name}  ({dt:.1f}s)  -> {type(exc).__name__}: {exc}")
    print("\nReading:")
    print("  200 = reachable.  Timeout/ConnectionError = blocked or too slow.")
    print("  If PDBe + RCSB downloads work but chemcomp/UniProt fail, a firewall")
    print("  is likely blocking those specific hosts.")


if __name__ == "__main__":
    main()
