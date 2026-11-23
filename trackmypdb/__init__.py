"""
TrackMyPDB: extract ligands (heteroatoms) from the PDB structures of UniProt
proteins and rank them by fingerprint similarity to a query molecule.

The package is a plain Python library with no Streamlit dependency. The
Streamlit web app (``streamlit_app.py``), the command-line interface
(``trackmypdb`` / ``python -m trackmypdb``) and the MCP server
(``mcp_server``) are thin layers on top of it.

Quick start::

    from trackmypdb import HeteroatomExtractor, rank_by_similarity

    ligands = HeteroatomExtractor().extract(["P37231"], max_structures=5).records
    hits = rank_by_similarity("CC(=O)Oc1ccccc1C(=O)O", ligands, top_n=10)

MIT License.
"""

__version__ = "2.1.0"

from .extraction import (  # noqa: E402
    ExtractionResult,
    HeteroatomExtractor,
    is_valid_uniprot_accession,
    parse_uniprot_ids,
)
from .ligand_db import load_ligand_db  # noqa: E402


def __getattr__(name):
    """Lazily expose RDKit-dependent helpers so that importing the package
    (and using the extractor) works even before RDKit is installed."""
    if name in {"rank_by_similarity", "compare_smiles", "fingerprint",
                "InvalidSmilesError", "FingerprintSettings"}:
        from . import similarity
        return getattr(similarity, name)
    raise AttributeError(f"module 'trackmypdb' has no attribute {name!r}")


__all__ = [
    "__version__",
    "ExtractionResult",
    "HeteroatomExtractor",
    "is_valid_uniprot_accession",
    "parse_uniprot_ids",
    "load_ligand_db",
    "rank_by_similarity",
    "compare_smiles",
    "fingerprint",
    "InvalidSmilesError",
    "FingerprintSettings",
]
