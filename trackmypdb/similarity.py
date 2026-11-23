"""
Fingerprint-based molecular similarity (Morgan/ECFP-like fingerprints and the
Tanimoto coefficient) using RDKit's ``rdFingerprintGenerator`` API.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")  # silence per-molecule parse warnings

SIMILARITY_COLUMN = "Tanimoto_Similarity"


class InvalidSmilesError(ValueError):
    """Raised when a query SMILES cannot be parsed."""


@dataclass(frozen=True)
class FingerprintSettings:
    radius: int = 2
    n_bits: int = 2048

    def __post_init__(self):
        if self.radius < 0:
            raise ValueError("radius must be >= 0")
        if self.n_bits <= 0:
            raise ValueError("n_bits must be positive")


@functools.lru_cache(maxsize=16)
def _generator(radius: int, n_bits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def mol_from_smiles(smiles: str | None):
    """Parse SMILES with RDKit. If full sanitisation fails (typically
    kekulisation problems in deposited ligand SMILES), retry with a partial
    sanitisation so the molecule can still be fingerprinted. Returns ``None``
    for unparseable input."""
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return None
    smiles = str(smiles).strip()
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            catchErrors=True,
        )
        Chem.GetSymmSSSR(mol)
        return mol
    except Exception:  # noqa: BLE001
        return None


def fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Morgan bit-vector fingerprint for a SMILES string, or ``None``."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return _generator(radius, n_bits).GetFingerprint(mol)
    except Exception:  # noqa: BLE001
        return None


def fingerprints(smiles: Iterable[str], radius: int = 2, n_bits: int = 2048) -> list:
    """Fingerprints for many SMILES (``None`` where parsing fails)."""
    return [fingerprint(s, radius, n_bits) for s in smiles]


def compare_smiles(smiles_a: str, smiles_b: str, radius: int = 2, n_bits: int = 2048) -> float:
    """Tanimoto similarity of two molecules. Raises ``InvalidSmilesError``."""
    fa, fb = fingerprint(smiles_a, radius, n_bits), fingerprint(smiles_b, radius, n_bits)
    if fa is None:
        raise InvalidSmilesError(f"Invalid SMILES: {smiles_a!r}")
    if fb is None:
        raise InvalidSmilesError(f"Invalid SMILES: {smiles_b!r}")
    return float(DataStructs.TanimotoSimilarity(fa, fb))


def bulk_tanimoto(query_fp, fps: Sequence) -> np.ndarray:
    """Tanimoto of ``query_fp`` against ``fps``; ``NaN`` where an entry is ``None``."""
    out = np.full(len(fps), np.nan)
    idx = [i for i, fp in enumerate(fps) if fp is not None]
    if idx:
        out[idx] = DataStructs.BulkTanimotoSimilarity(query_fp, [fps[i] for i in idx])
    return out


def rank_by_similarity(target_smiles: str, ligands: pd.DataFrame, smiles_column: str = "SMILES",
                       top_n: int | None = 50, min_similarity: float = 0.0,
                       radius: int = 2, n_bits: int = 2048,
                       precomputed_fps: Sequence | None = None) -> pd.DataFrame:
    """Rank ligand records by Tanimoto similarity to ``target_smiles``.

    All input columns are kept and a ``Tanimoto_Similarity`` column is added.
    Rows whose SMILES cannot be parsed are dropped. Results are sorted from
    most to least similar, filtered by ``min_similarity`` and truncated to
    ``top_n`` (``None`` returns every match).

    ``precomputed_fps`` (one fingerprint per row of ``ligands``, computed with
    the same settings) avoids recomputation when the same library is searched
    repeatedly.
    """
    FingerprintSettings(radius, n_bits)  # validates
    query_fp = fingerprint(target_smiles, radius, n_bits)
    if query_fp is None:
        raise InvalidSmilesError(f"Invalid target SMILES: {target_smiles!r}")

    df = ligands.reset_index(drop=True)
    if df.empty or smiles_column not in df.columns:
        return df.assign(**{SIMILARITY_COLUMN: pd.Series(dtype=float)}).iloc[0:0]

    fps = list(precomputed_fps) if precomputed_fps is not None else fingerprints(
        df[smiles_column].tolist(), radius, n_bits)
    if len(fps) != len(df):
        raise ValueError("precomputed_fps must have one entry per ligand row")

    scores = bulk_tanimoto(query_fp, fps)
    out = df.assign(**{SIMILARITY_COLUMN: scores})
    out = out[out[SIMILARITY_COLUMN].notna() & (out[SIMILARITY_COLUMN] >= min_similarity)]
    out = out.sort_values(SIMILARITY_COLUMN, ascending=False, kind="mergesort")
    if top_n:
        out = out.head(top_n)
    return out.reset_index(drop=True)
