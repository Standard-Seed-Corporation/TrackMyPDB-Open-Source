"""
TrackMyPDB MCP server (stdio transport).

Exposes the TrackMyPDB bioinformatics pipeline as Model Context Protocol tools
so ANY MCP client -- Claude Desktop, Cursor, or the in-app Claude agent -- can
call them. Built with the official MCP Python SDK (FastMCP).

Run manually:      python -m mcp_server.server
Claude Desktop:    see claude_desktop_config.example.json

MIT License - Open Source Project.
"""

from __future__ import annotations

import os
import sys

# Make sure the repo root is importable when launched as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp_server import core  # noqa: E402

mcp = FastMCP("trackmypdb")


@mcp.tool()
def list_pdbs_for_uniprot(uniprot_id: str) -> dict:
    """List the PDB structure IDs associated with a UniProt accession (e.g. P37231)."""
    return {"uniprot_id": uniprot_id, "pdb_ids": core.get_pdbs_for_uniprot(uniprot_id)}


@mcp.tool()
def extract_heteroatoms(uniprot_ids: list, max_pdbs_per_uniprot: int = 5) -> dict:
    """
    Extract drug-like heteroatoms (ligands) and their SMILES from the PDB
    structures of one or more UniProt proteins. Returns the ligand records plus
    a summary. Keep max_pdbs_per_uniprot small (<=5) to stay responsive.
    """
    return core.extract_heteroatoms(uniprot_ids, max_pdbs_per_uniprot=max_pdbs_per_uniprot)


@mcp.tool()
def analyze_similarity(target_smiles: str, heteroatoms: list,
                       top_n: int = 10, min_similarity: float = 0.0) -> dict:
    """
    Rank a list of heteroatom records (from extract_heteroatoms) by Tanimoto
    similarity to a target molecule given as SMILES.
    """
    return core.analyze_similarity(target_smiles, heteroatoms,
                                   top_n=top_n, min_similarity=min_similarity)


@mcp.tool()
def run_pipeline(uniprot_ids: list, target_smiles: str,
                 max_pdbs_per_uniprot: int = 5, top_n: int = 10,
                 min_similarity: float = 0.0) -> dict:
    """
    Full pipeline: extract heteroatoms for the given UniProt IDs, then rank them
    by similarity to target_smiles. Prefer this when the user gives both protein
    IDs and a target molecule.
    """
    return core.run_pipeline(uniprot_ids, target_smiles,
                             max_pdbs_per_uniprot=max_pdbs_per_uniprot,
                             top_n=top_n, min_similarity=min_similarity)


@mcp.tool()
def compare_smiles(smiles_a: str, smiles_b: str) -> dict:
    """Compute the Tanimoto similarity (0-1) between two molecules given as SMILES."""
    return core.compare_smiles(smiles_a, smiles_b)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
