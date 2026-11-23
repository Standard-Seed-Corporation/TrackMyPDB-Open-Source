# JOSS submission checklist (internal)

This lists what is already in the repository for a Journal of Open Source Software
submission and what the team still has to do. Requirements are taken from the JOSS
author guide (https://joss.readthedocs.io), checked in November 2026. Re-check the
guide before submitting, as it changes.

## Already in place (v2.1.0)

- [x] OSI-approved license (`LICENSE`, MIT)
- [x] Core library separate from the web app. JOSS only accepts web tools that expose
      a core library through the interface, and excludes "thin" API clients.
- [x] Packaging to community standards (`pyproject.toml`, `pip install .`), CLI entry point
- [x] Automated tests (`tests/`) and CI on three operating systems (`.github/workflows/tests.yml`)
- [x] Installation, usage examples and API overview in `README.md`
- [x] Community guidelines: `CONTRIBUTING.md` (how to contribute, report issues and get
      support), `CODE_OF_CONDUCT.md`, issue and PR templates
- [x] `CITATION.cff` and `CHANGELOG.md`
- [x] `paper/paper.md` with the required sections (Summary, Statement of need, State of the
      field, Software design, Research impact statement, AI usage disclosure) and
      `paper/paper.bib`; `.github/workflows/draft-pdf.yml` builds the PDF preview

## To do before submitting

### Paper (`paper/paper.md`)
- [ ] Fill every `TODO` comment: ORCIDs, affiliations (with ROR IDs), corresponding
      author, submission date, funding statement.
- [ ] Write the **Research impact statement** with *evidence* of research use
      (publications, theses, analyses, web-app usage numbers, external users).
      Intended future use is not enough.
- [ ] Rewrite the **Statement of need** in your own words around the real research problem.
- [ ] Complete the **AI usage disclosure**: list every AI tool and model version used for
      code, docs and paper, and confirm human review. Incomplete disclosure can lead to
      desk rejection.
- [ ] Verify every reference DOI in `paper.bib`.
- [ ] Keep it between 750 and 1,750 words (currently about 900 words; the TODO sections will add to this).
- [ ] Check the rendered PDF from the "Draft JOSS paper PDF" workflow.

### Repository and project history
- [ ] Make the repository public on GitHub with issues enabled. JOSS screens for **at
      least six months of public development history** with activity spread over that
      period (not one large burst of commits), plus releases, issues and pull requests.
      If the history is shorter, keep developing in the open and submit later.
- [ ] Commit this refactor as a series of logical commits or pull requests, not one
      giant commit, and use issues/PRs for further work from now on.
- [ ] Replace `<org>` / `<repository-url>` placeholders in `README.md`, `CITATION.cff`,
      `pyproject.toml` and `CONTRIBUTING.md`.
- [ ] Add a maintainer contact e-mail in `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.
- [ ] Resolve the copyright holder mismatch: `LICENSE` says "2024 Anu Gamage", the README
      previously said "© 2025 Standard Seed Corporation, all rights reserved".
      Agree on one holder (e.g. "2024-2026 The TrackMyPDB developers") and use it everywhere.
- [ ] Tag a release (`v2.1.0`), let CI pass, and archive it on Zenodo to get a DOI
      (needed at acceptance).
- [ ] Consider publishing to PyPI (`pip install trackmypdb`) to make reviewer installation trivial.
- [ ] Ask a colleague outside the team to install and run it from the README alone.

### Submission
- [ ] Every author has a GitHub account; the submitting author is a major contributor.
- [ ] Submit at https://joss.theoj.org/papers/new with the repository URL and the
      `paper/` branch path.
- [ ] During review, respond to reviewers yourselves (JOSS does not allow AI-generated
      replies to editors or reviewers, except for translation).
