# Contributing to TrackMyPDB

Thank you for your interest in improving TrackMyPDB! Contributions of all kinds are
welcome: bug reports, feature ideas, documentation fixes and code.

## Reporting problems and asking for help

- **Bugs and feature requests:** open an issue on GitHub using the templates. Include the
  UniProt IDs, SMILES and settings you used, and the TrackMyPDB version
  (`trackmypdb --version`).
- **Questions about usage:** open an issue with the `question` label.
- **Security issues** (for example, a way to expose API keys of a deployment): please
  contact the maintainers privately instead of opening a public issue.
  <!-- TODO: add a maintainer contact e-mail -->

We aim to respond to new issues within two weeks.

## Development setup

```bash
git clone <repository-url>
cd TrackMyPDB-Open-Source
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[app,agent,test]"
pytest                    # offline test suite
pytest --run-network      # also run live API tests (slower)
streamlit run streamlit_app.py
```

## Project layout

| Path | Contents |
|------|----------|
| `trackmypdb/` | The library: extraction, similarity, annotations, CLI. **No Streamlit code here.** |
| `ui/`, `streamlit_app.py` | Streamlit pages (user interface only) |
| `mcp_server/`, `agent/` | MCP server and in-app AI assistant |
| `tests/` | pytest suite; `tests/data/` holds small structure fixtures |
| `paper/` | JOSS paper (`paper.md`, `paper.bib`) |

## Making a change

1. Fork the repository and create a branch from `main`.
2. Put scientific logic in `trackmypdb/` and keep the UI layers thin.
3. Add or update tests. Tests must not depend on the network unless they are
   marked `@pytest.mark.network`; use the `FakeSession` fixture in `tests/conftest.py`
   to simulate API responses.
4. Update the documentation and add a line to `CHANGELOG.md` under *Unreleased*.
5. Open a pull request and describe the motivation. CI must pass before merging.

Code style: follow PEP 8, use type hints for public functions and write docstrings
that say what a function returns. Keep line length at 100 characters.

## Releases

Maintainers tag releases as `vX.Y.Z` following semantic versioning, update
`trackmypdb/__version__`, `CITATION.cff` and `CHANGELOG.md`, and archive each release
on Zenodo.

## Code of conduct

By participating you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).
