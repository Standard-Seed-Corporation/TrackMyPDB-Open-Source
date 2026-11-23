"""
TrackMyPDB MCP server (stdio transport).

Exposes the TrackMyPDB library as Model Context Protocol tools so that any MCP
client (Claude Desktop, Cursor, the in-app assistant...) can call them.

Run:  python -m mcp_server.server
MIT License.
"""

from __future__ import annotations

import logging
import os
import sys

# Make the repository root importable when launched as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# stdout carries the MCP protocol: send all logging to stderr.
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

from mcp.server.fastmcp import FastMCP  # noqa: E402

from mcp_server import core  # noqa: E402

mcp = FastMCP("trackmypdb")


@mcp.tool()
def list_pdbs_for_uniprot(uniprot_id: str) -> dict:
    """List PDB structure IDs for a UniProt accession (e.g. P37231), best structures first."""
    return {"uniprot_id": uniprot_id, "pdb_ids": core.get_pdbs_for_uniprot(uniprot_id)}


@mcp.tool()
def extract_heteroatoms(uniprot_ids: list[str], max_pdbs_per_uniprot: int = 5) -> dict:
    """Extract ligands (heteroatoms) and their SMILES from the PDB structures of
    one or more UniProt proteins. Water, ions and buffers are excluded. Keep
    max_pdbs_per_uniprot small (<=5) to stay responsive."""
    return core.extract_heteroatoms(uniprot_ids, max_pdbs_per_uniprot=max_pdbs_per_uniprot)


@mcp.tool()
def analyze_similarity(target_smiles: str, heteroatoms: list[dict], top_n: int = 10,
                       min_similarity: float = 0.0) -> dict:
    """Rank heteroatom records (from extract_heteroatoms) by Tanimoto similarity
    (Morgan fingerprints, radius 2, 2048 bits) to a target SMILES."""
    return core.analyze_similarity(target_smiles, heteroatoms, top_n=top_n,
                                   min_similarity=min_similarity)


@mcp.tool()
def run_pipeline(uniprot_ids: list[str], target_smiles: str, max_pdbs_per_uniprot: int = 5,
                 top_n: int = 10, min_similarity: float = 0.0) -> dict:
    """Extract ligands for UniProt IDs, then rank them by similarity to
    target_smiles. Prefer this when the user gives both proteins and a molecule."""
    return core.run_pipeline(uniprot_ids, target_smiles, max_pdbs_per_uniprot=max_pdbs_per_uniprot,
                             top_n=top_n, min_similarity=min_similarity)


@mcp.tool()
def search_ligand_database(target_smiles: str, top_n: int = 10,
                           min_similarity: float = 0.3) -> dict:
    """Find PDB ligands similar to target_smiles in the bundled offline ligand
    database (fast, no UniProt ID needed)."""
    return core.search_ligand_database(target_smiles, top_n=top_n, min_similarity=min_similarity)


@mcp.tool()
def compare_smiles(smiles_a: str, smiles_b: str) -> dict:
    """Tanimoto similarity (0-1) between two molecules given as SMILES."""
    return core.compare_smiles(smiles_a, smiles_b)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
