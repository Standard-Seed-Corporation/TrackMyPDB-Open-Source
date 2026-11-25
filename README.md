# TrackMyPDB

<!-- TODO: replace <org> with the GitHub organisation once the repository URL is final. -->
[![tests](https://github.com/<org>/TrackMyPDB-Open-Source/actions/workflows/tests.yml/badge.svg)](https://github.com/<org>/TrackMyPDB-Open-Source/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Web app](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://trackmypdbsscai.streamlit.app/)

**Find the ligands that have been co-crystallised with your proteins in the Protein Data
Bank, and rank them by chemical similarity to a molecule of interest.**

Given UniProt accessions, TrackMyPDB:

1. maps them to PDB entries (PDBe best-structure mapping, best structures first);
2. downloads each structure (mmCIF, falling back to PDB format) and extracts its bound
   chemical components, dropping water, ions, buffers, cryo-protectants and modified
   residues;
3. resolves each component to SMILES, name, formula and InChIKey from the wwPDB Chemical
   Component Dictionary (with an offline fallback database);
4. ranks the ligands by Tanimoto similarity of Morgan fingerprints (RDKit) to one or
   more query molecules;
5. optionally annotates hits with UniProt accessions, protein names and UniProt disease
   involvement.

It also supports substructure (scaffold) search, highlights the atoms each hit shares
with the query, and shows any hit in its 3D binding site.

It is one Python library with four interfaces: a **Python API**, a **command-line
tool**, a **Streamlit web app** and an **MCP server** for AI assistants.

## Installation

Python 3.10 or newer is required.

```bash
git clone <repository-url>
cd TrackMyPDB-Open-Source
pip install -e .                 # library + command-line tool
pip install -e ".[app]"          # + web app
pip install -e ".[app,agent]"    # + AI assistant / MCP server
```

RDKit is installed from PyPI automatically. With conda you can instead use
`conda install -c conda-forge rdkit` before installing TrackMyPDB.

Windows users can double-click `install.bat` (creates a virtual environment) and then
`run.bat` to start the web app.

## Usage

### Web app

```bash
streamlit run streamlit_app.py
```

Then open <http://localhost:8501>. A hosted version is available at
<https://trackmypdbsscai.streamlit.app/>.

| Page | What it does |
|------|--------------|
| Quick Search | Similarity or substructure (scaffold) search in the built-in ligand library (offline, instant) |
| Target Search | UniProt IDs + query SMILES → ligands from your proteins' live PDB structures, ranked |
| Ligand Extraction | Ligand inventory of your proteins' PDB structures |
| Similarity Ranking | Rank extracted or uploaded ligands against a query |
| Molecule Viewer | 2D depiction and drug-likeness descriptors |
| Disease Enrichment | UniProt disease involvement of the protein targets |
| AI Assistant | Ask questions in plain language (needs an Anthropic API key) |

### Command line

```bash
# Ligands bound in the 5 best structures of PPAR-gamma
trackmypdb extract P37231 --max-structures 5 -o ppar_ligands.csv

# Rank them against a query molecule (rosiglitazone)
trackmypdb similarity "CN(CCOc1ccc(CC2SC(=O)NC2=O)cc1)c1ccccn1" ppar_ligands.csv --top 10

# Both steps at once
trackmypdb pipeline P37231 --smiles "CN(CCOc1ccc(CC2SC(=O)NC2=O)cc1)c1ccccn1" -o hits.csv

# Every ligand containing a quinazoline scaffold (SMILES or SMARTS)
trackmypdb substructure "c1ccc2ncncc2c1" --top 20

# Search the bundled ligand database (no network needed)
trackmypdb search "CC(=O)Oc1ccccc1C(=O)O" --min-similarity 0.4

# Similarity of two molecules
trackmypdb compare "CC(=O)Oc1ccccc1C(=O)O" "Oc1ccccc1C(=O)O"
```

Run `trackmypdb <command> --help` for all options.

### Python

```python
from trackmypdb import HeteroatomExtractor, rank_by_similarity
from trackmypdb.annotations import add_protein_info

result = HeteroatomExtractor().extract(["P37231"], max_structures=5)
print(result.summary)                  # counts, failed downloads, invalid IDs
ligands = result.records               # pandas DataFrame, one row per protein/entry/ligand

hits = rank_by_similarity("CN(CCOc1ccc(CC2SC(=O)NC2=O)cc1)c1ccccn1", ligands, top_n=10)
hits = add_protein_info(hits)          # adds UniProt_IDs and Protein_Names
```

### AI assistant and MCP server

See [docs/AGENTIC_MCP.md](docs/AGENTIC_MCP.md). The MCP server (`python -m mcp_server.server`)
works with any MCP client, such as Claude Desktop.

## Output columns

| Column | Meaning |
|--------|---------|
| `UniProt_ID`, `PDB_ID` | Protein accession and PDB entry |
| `Heteroatom_Code` | Chemical Component Dictionary code of the ligand |
| `Chemical_Name`, `SMILES`, `Formula`, `InChIKey` | Ligand chemistry |
| `Component_Type` | CCD component type (e.g. `NON-POLYMER`) |
| `Source` | Where the chemistry came from: `rcsb`, `rcsb_sdf`, `local_db` or `not_found` |
| `Chains`, `Residue_Numbers`, `Copies`, `Atom_Count` | Where and how often it occurs in the entry |
| `Tanimoto_Similarity` | Similarity to the query (0–1), in ranking results |

## Testing

```bash
pip install -e ".[test]"
pytest                    # offline suite, uses recorded API responses
pytest --run-network      # also checks the live RCSB / PDBe / UniProt APIs
```

Tests run automatically on GitHub Actions for Linux, Windows and macOS.

## Limitations

- Extraction depends on public web services (PDBe, RCSB PDB, UniProt); large
  proteins can map to hundreds of structures, so use `--max-structures` for quick looks.
- Similarity is 2D fingerprint similarity. It does not consider binding mode,
  stereochemistry (by default) or activity.
- Disease annotations are limited to the curated UniProtKB/Swiss-Prot "Involvement in
  disease" section.

## Contributing and support

Bug reports, questions and pull requests are welcome. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md), and use
GitHub issues for support.

## Citation

If you use TrackMyPDB, please cite it using the metadata in [CITATION.cff](CITATION.cff)
(GitHub's "Cite this repository" button gives APA and BibTeX). Please also acknowledge
the RCSB PDB, PDBe, UniProt and RDKit.

## License

MIT, see [LICENSE](LICENSE). Please respect the terms of use of the RCSB PDB, PDBe and
UniProt APIs.

## Authors

Developed and released by Standard Seed Corporation.

- Project supervisor: [Suliman Sharif](https://www.linkedin.com/in/sulimansharif/)
- Lead engineer: [Anu Gamage](https://www.linkedin.com/in/anu-gamage-62192b201/)
- Associate engineers: [Kalana Kotawalagedara](https://www.linkedin.com/in/kalana-kotawalagedara-962939225/),
  [Sakeer MR](https://www.linkedin.com/in/mohammadhu-rifaydheen-sakeer/),
  [Damilola Bodun](https://www.linkedin.com/in/damilola-bodun-123987208/)
