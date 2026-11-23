---
title: 'TrackMyPDB: linking ligands in protein structures to query molecules by fingerprint similarity'
tags:
  - Python
  - cheminformatics
  - structural biology
  - Protein Data Bank
  - molecular similarity
  - drug discovery
authors:
  # TODO: confirm author order, add every author's ORCID, mark the corresponding author.
  - name: Suliman Sharif
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Anu Gamage
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: 1
  - name: Kalana Kotawalagedara
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Sakeer Sha
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Damilola Bodun
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  # TODO: add university affiliations (and ROR IDs) where applicable.
  - name: Standard Seed Corporation
    index: 1
date: 23 November 2026  # TODO: set to the submission date
bibliography: paper.bib
---

# Summary

Many questions in drug discovery and chemical biology start from a protein and a
molecule: *which compounds have already been observed bound to this target, and how
closely do they resemble the molecule I am working on?* The Protein Data Bank (PDB)
[@Berman2003; @Burley2023] holds this information, but it is scattered across
hundreds of structure files per well-studied protein, recorded as three-letter
chemical-component codes rather than as searchable chemical structures, and mixed with
water, ions, buffers and crystallisation additives.

TrackMyPDB is a Python package that turns this into a single reproducible step. Given
UniProt accessions [@UniProt2023], it retrieves every mapped PDB entry through the PDBe
best-structure mapping [@Armstrong2020], extracts the bound chemical components from
the mmCIF coordinate files, discards solvent, ions, common additives and
polymer-linking residues, and resolves each remaining component to SMILES, name,
formula and InChIKey using the wwPDB Chemical Component Dictionary [@Westbrook2015]. The
resulting ligand table can then be ranked against one or more query molecules by
Tanimoto similarity [@Bajusz2015] of Morgan (ECFP-like) fingerprints [@Rogers2010]
computed with RDKit [@RDKit], and annotated with the UniProt accessions, protein names
and curated disease involvement of the matching targets.

The same functionality is available through four interfaces built on one library: a
Python API, a command-line tool, a Streamlit [@Streamlit] web application for users who
do not program, and a Model Context Protocol server [@MCP] that lets AI assistants call
the pipeline as tools. A bundled offline database of ligands from more than 2,500 PDB
entries supports fast similarity search without network access.

# Statement of need

<!-- TODO (authors): describe the concrete research problem that motivated the
software, in your own words, and who uses it (e.g. medicinal chemists checking
whether a scaffold has structural precedent on a target, structural biologists
cataloguing cofactors across a protein family, students exploring target-ligand
relationships). -->

Answering "what is known to bind my protein, and what resembles my compound?" usually
requires stitching together several resources by hand: mapping a UniProt accession to
PDB entries, downloading and parsing each structure, deciding which heteroatoms are
genuine ligands, looking up each component's chemistry, and finally running a
similarity calculation in a separate cheminformatics environment. Each step is simple
but the combination is tedious, error-prone and rarely recorded in a reproducible way.
Typical failure modes include silently missing large cryo-EM entries that are only
distributed as mmCIF, counting selenomethionine or glycerol as ligands, and mismatching
component codes with unrelated compounds of the same name in general chemical databases.

TrackMyPDB is aimed at researchers who need this target-centred ligand view
repeatedly and transparently. It records where every SMILES came from, reports which
structures could not be retrieved, and exposes all parameters (number of structures per
protein, exclusion lists, fingerprint radius and length, similarity threshold), so that
a search can be rerun and reported precisely.

# State of the field

Large curated resources describe protein–ligand complexes, notably BioLiP2
[@Zhang2024], which annotates biologically relevant ligand–protein interactions across
the PDB, and ChEMBL [@Zdrazil2024], which links compounds to targets through bioactivity
measurements rather than structures. The RCSB PDB web site offers chemical similarity
search over chemical components, and libraries such as Biopython [@Cock2009] provide
structure parsing, while PLIP [@Adasme2021] profiles interactions within individual
complexes. TrackMyPDB is complementary to these: it starts from a user-defined set of
proteins, works directly on the live PDB and Chemical Component Dictionary, combines
the ligand inventory with fingerprint ranking and target/disease annotation in one
scriptable pipeline, and returns plain tables that can be passed to these tools for
deeper analysis. Rather than re-implementing chemistry, it builds on RDKit and pandas
[@McKinney2010].

<!-- TODO (authors): if you know of other tools with overlapping functionality,
add them here with an honest comparison; JOSS reviewers check this section. -->

# Software design

The core is the `trackmypdb` package, which has no user-interface dependency.
`HeteroatomExtractor` performs the extraction in four stages: (1) accession-to-entry
mapping, kept in PDBe's quality order so that a per-protein limit selects the best
structures; (2) parallel download and parsing of coordinate files, preferring mmCIF and
falling back to the legacy PDB format; (3) resolution of each *unique* component code
exactly once, through the RCSB chemical-component service, then the RCSB
ideal-coordinate SDF, then the bundled offline database; and (4) filtering and
assembly of one record per protein, entry and ligand, including chain identifiers,
copy number and the provenance of the chemical data. Network access goes through a
shared session with retries and back-off, and failures are reported in the
`ExtractionResult` instead of aborting a run.

`rank_by_similarity` operates on any table with a SMILES column, so it can rank
extraction output, the bundled database or user-supplied libraries. Fingerprints use
RDKit's `rdFingerprintGenerator` API and can be precomputed once for repeated searches.
Deposited ligand SMILES that fail full sanitisation are retried with partial
sanitisation rather than silently dropped. The `substructure` module adds scaffold searches with
SMILES or SMARTS queries and computes maximum common substructures, which the
interfaces use to highlight the fragment a hit shares with the query. The `annotations`
module adds protein information through batched RCSB GraphQL queries and disease
involvement from UniProtKB.

The command-line interface, the Streamlit application and the MCP server are thin
layers over this API; the web application additionally displays each hit in its
binding site with 3Dmol.js. This separation keeps the scientific logic testable: the test
suite exercises extraction, parsing, fallback logic and annotation against recorded
API responses, runs without network access in continuous integration, and includes
optional live tests that detect upstream API changes.

# Research impact statement

<!-- TODO (authors) - REQUIRED. JOSS needs *evidence* of research use, not
intended use. Replace this comment with concrete, verifiable items, e.g.:
  - projects, theses or publications that used TrackMyPDB (with citations),
  - analyses run by the developers for real research questions,
  - usage of the public web app (number of users/sessions over a period),
  - external users, issues or contributions on GitHub.
Do not present new research results here. -->

# AI usage disclosure

<!-- TODO (authors): complete and verify. JOSS requires this section to name the
tools and model versions, where they were used, and to confirm human review. -->

Generative AI tools were used during development of this software and paper. Version
2.1 of the code base (the refactoring into a standalone library, the command-line
interface, the test suite and the restructured web application), the documentation and
an initial draft of this paper were prepared with assistance from Claude Opus 5.5
(Anthropic). The in-application AI assistant uses Anthropic Claude models at run time.
The authors reviewed, edited and tested all AI-assisted code and text, verified the
references, and made the design decisions described above.
<!-- TODO: add any other AI tools used earlier in the project (e.g. code assistants
used for versions 1.x-2.0), with model versions. -->

# Acknowledgements

We acknowledge the RCSB PDB, PDBe and the wider wwPDB partnership, UniProt and the
RDKit community for the open data and software on which TrackMyPDB depends.
<!-- TODO: acknowledge funding sources (required by JOSS if any), or state that the
work received no specific funding. -->

# References
