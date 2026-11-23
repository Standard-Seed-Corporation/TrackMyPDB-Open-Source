# Changelog

All notable changes to TrackMyPDB are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- Substructure search (SMILES or SMARTS scaffolds) in the library
  (`trackmypdb.substructure`), the CLI (`trackmypdb substructure`) and the
  Database Search page, with example scaffolds.
- Maximum-common-substructure highlighting: hit cards shade the atoms each ligand
  shares with the query.
- Interactive 3D binding-site viewer (3Dmol.js): ligand, surrounding residues within
  5 Å and a link to the RCSB 3D view, on all result pages.
- Redesigned interface: home page with banner, live "Try it" search with example
  drugs, structure cards with similarity badges, page banners, example loaders and a
  live query preview with drug-likeness descriptors.

## [2.1.0] - 2026-11-23

### Fixed
- Similarity Ranking and the pipeline returned at most **10** results regardless of the
  "Number of results" setting, and the "complete" CSV download contained only those 10
  rows. All matching results up to the chosen limit are now returned.
- Extraction crashed with `KeyError: 'PDB_ID'` when no structures could be retrieved
  (invalid or unmapped UniProt IDs, network failure). It now returns an empty table
  and reports which IDs were invalid or had no structures.
- Results were written to shared files in the app folder (`heteroatom_results.csv`, …),
  so on a hosted deployment one user's results could appear for another user, and
  "Clear results" deleted them for everyone. Results now live only in each user's session.
- Large structures that are only distributed as mmCIF (most cryo-EM entries) always
  failed to download. Structures are now read from mmCIF, with PDB format as fallback.
- The PubChem fallback looked up three-letter PDB component codes as compound *names*,
  which could silently return an unrelated molecule. It was removed; the RCSB
  ideal-coordinate SDF and the bundled database are used instead.
- Modified residues (e.g. selenomethionine) and many common additives were reported as
  ligands. Polymer-linking components are now excluded and the exclusion list was extended.
- The "no RDKit" fallback analyzer could never load (RDKit was already imported by
  another module) and produced meaningless hash-based scores. It was removed; RDKit is
  a required dependency.
- `Lipinski_Violations` was referenced but no longer calculated.
- The AI assistant could loop without limit, kept a broken history after an error, and
  silently used the deployment's API key for every visitor.
- `python-dotenv` was installed but `.env` was never loaded.

### Changed
- **New `trackmypdb` library** with no Streamlit dependency (`extraction`, `similarity`,
  `ligand_db`, `annotations`, `properties`, `cli`). The web app, CLI and MCP server
  are thin layers on top of it. `backend/` and `mcp_server/st_shim.py` were removed.
- Extraction downloads structures in parallel and resolves each unique ligand only once
  (previously every ligand was looked up again for every structure, with a fixed
  delay), and the UniProt→PDB mapping is requested once instead of twice.
- PDB entries are kept in PDBe's quality order, so a per-protein limit uses the best
  structures rather than the alphabetically first ones.
- Fingerprints use RDKit's `rdFingerprintGenerator` API (the deprecated
  `GetMorganFingerprintAsBitVect` call and the global warning suppression were removed).
- Database Search fingerprints the bundled database once per setting and caches it;
  previously every query recomputed all ~3,500 fingerprints.
- Protein information is fetched with batched RCSB GraphQL queries, disease
  annotations with parallel UniProt requests.
- Web app rebuilt with Streamlit's native multipage navigation; pages live in `ui/`,
  duplicated code was merged into shared components, a debug caption was removed,
  and Similarity Ranking accepts uploaded CSV files.
- Streamlit CORS and XSRF protection are no longer disabled.
- New output columns: `Component_Type`, `InChIKey`, `Source` (replaces `Status`),
  `Copies`. The bundled database column `Status` is read as `Component_Type`.

### Added
- Command-line interface: `trackmypdb extract | similarity | search | pipeline | compare`.
- `pyproject.toml` packaging (`pip install -e .`) with optional extras `app`, `agent`, `test`.
- Offline pytest suite with recorded API responses, optional live API tests, and GitHub
  Actions CI on Linux, Windows and macOS.
- MCP tool `search_ligand_database` (offline).
- JOSS paper draft (`paper/`), `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  issue and pull-request templates.

### Removed
- Status notes (`PROJECT_SUMMARY.md`, `ENHANCEMENT_STATUS.md`,
  `IMPLEMENTATION_SUMMARY.md`, `TESTING_GUIDE.md`, `STARTUP_GUIDE.md`), the empty
  `example_inputs.md`, ad-hoc test scripts, committed result CSVs, `__pycache__`
  folders, the redistributed third-party PDF on software citation, and the Claude Desktop
  config example containing a personal path (now documented in `docs/AGENTIC_MCP.md`).

## [2.0.0] - 2025

Initial public release of the Streamlit application (heteroatom extraction, similarity
analysis, SMILES database search, molecule visualizer, disease enrichment, AI assistant).
