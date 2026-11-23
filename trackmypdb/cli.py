"""
Command-line interface.

Examples::

    trackmypdb extract P37231 Q9UNQ0 --max-structures 5 -o ligands.csv
    trackmypdb similarity "CC(=O)Oc1ccccc1C(=O)O" ligands.csv --top 20
    trackmypdb search "CC(=O)Oc1ccccc1C(=O)O" --min-similarity 0.4
    trackmypdb pipeline P37231 --smiles "CC(=O)Oc1ccccc1C(=O)O" -o hits.csv
    trackmypdb substructure "c1ccc2ncncc2c1" --top 20
    trackmypdb compare CCO CCN
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from . import __version__


def _write(df: pd.DataFrame, path: str | None) -> None:
    if path:
        df.to_csv(path, index=False)
        print(f"Wrote {len(df)} rows to {path}", file=sys.stderr)
    else:
        df.to_csv(sys.stdout, index=False)


def _progress(frac: float, msg: str) -> None:
    print(f"\r[{frac:6.1%}] {msg:<60}", end="", file=sys.stderr, flush=True)
    if frac >= 1.0:
        print(file=sys.stderr)


def _fp_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--top", type=int, default=50, help="number of hits to keep (0 = all)")
    p.add_argument("--min-similarity", type=float, default=0.0)
    p.add_argument("--radius", type=int, default=2, help="Morgan radius")
    p.add_argument("--bits", type=int, default=2048, help="fingerprint length")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trackmypdb", description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="extract ligands for UniProt accessions")
    p.add_argument("uniprot_ids", nargs="+")
    p.add_argument("--max-structures", type=int, default=0,
                   help="max PDB entries per accession, best first (0 = all)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--keep-all", action="store_true",
                   help="keep water/ions/buffers and polymer components")
    p.add_argument("-o", "--output")

    p = sub.add_parser("similarity", help="rank a ligand CSV against a query SMILES")
    p.add_argument("smiles")
    p.add_argument("ligands_csv")
    _fp_args(p)
    p.add_argument("-o", "--output")

    p = sub.add_parser("search", help="search the bundled PDB ligand database")
    p.add_argument("smiles")
    _fp_args(p)
    p.add_argument("-o", "--output")

    p = sub.add_parser("substructure",
                       help="find ligands containing a scaffold (SMILES or SMARTS)")
    p.add_argument("pattern")
    p.add_argument("ligands_csv", nargs="?",
                   help="ligand CSV to search (default: the bundled ligand database)")
    p.add_argument("--top", type=int, default=50, help="max matches (0 = all)")
    p.add_argument("-o", "--output")

    p = sub.add_parser("pipeline", help="extract + rank in one step")
    p.add_argument("uniprot_ids", nargs="+")
    p.add_argument("--smiles", required=True)
    p.add_argument("--max-structures", type=int, default=0)
    _fp_args(p)
    p.add_argument("-o", "--output")

    p = sub.add_parser("compare", help="Tanimoto similarity of two molecules")
    p.add_argument("smiles_a")
    p.add_argument("smiles_b")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--bits", type=int, default=2048)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    if args.command in ("extract", "pipeline"):
        from .extraction import EXCLUDED_CODES, HeteroatomExtractor
        keep_all = getattr(args, "keep_all", False)
        extractor = HeteroatomExtractor(
            workers=getattr(args, "workers", 6),
            exclude_codes=frozenset() if keep_all else EXCLUDED_CODES,
            exclude_polymer_components=not keep_all)
        result = extractor.extract(args.uniprot_ids, max_structures=args.max_structures or None,
                                   progress=_progress)
        for key, value in result.summary.items():
            print(f"{key}: {value}", file=sys.stderr)
        if args.command == "extract":
            _write(result.records, args.output)
            return 0
        ligands = result.records

    from .similarity import InvalidSmilesError, compare_smiles, rank_by_similarity

    try:
        if args.command == "substructure":
            from .ligand_db import load_ligand_db
            from .substructure import substructure_search
            ligands = (pd.read_csv(args.ligands_csv, dtype=str, keep_default_na=False)
                       if args.ligands_csv else load_ligand_db(with_smiles_only=True))
            _write(substructure_search(args.pattern, ligands, max_results=args.top or None),
                   args.output)
            return 0
        if args.command == "compare":
            print(f"{compare_smiles(args.smiles_a, args.smiles_b, args.radius, args.bits):.4f}")
            return 0
        if args.command == "similarity":
            ligands = pd.read_csv(args.ligands_csv, dtype=str, keep_default_na=False)
        elif args.command == "search":
            from .ligand_db import load_ligand_db
            ligands = load_ligand_db(with_smiles_only=True)
        hits = rank_by_similarity(args.smiles, ligands, top_n=args.top or None,
                                  min_similarity=args.min_similarity,
                                  radius=args.radius, n_bits=args.bits)
    except InvalidSmilesError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _write(hits, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
