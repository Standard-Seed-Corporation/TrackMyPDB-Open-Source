# TrackMyPDB - Agentic Layer & MCP Server

This feature turns TrackMyPDB from a click-through Streamlit app into something
users can **talk to**. It has two parts that share one set of tools:

1. **MCP server** (`mcp_server/`) - exposes the TrackMyPDB pipeline as
   [Model Context Protocol](https://modelcontextprotocol.io) tools over stdio.
   Any MCP client (Claude Desktop, Cursor, ...) can use it.
2. **Agentic chat tab** (`agent/`) - a new "AI Assistant" page in the Streamlit
   app. It uses Claude, which calls the very same MCP server to answer.

```
                       +---------------------------+
   Streamlit "AI       |   agent/mcp_agent.py      |
   Assistant" tab  --> |   (Claude tool-use loop)  | --+
                       +---------------------------+   |
                                                       |  stdio (MCP)
   Claude Desktop / Cursor  ----------------------------+--> mcp_server/server.py
                                                              |
                                                       mcp_server/core.py
                                                              |
                                       backend/heteroatom_extractor.py
                                       backend/similarity_analyzer.py
```

## Tools exposed

| Tool | What it does |
|------|--------------|
| `list_pdbs_for_uniprot` | PDB IDs for a UniProt accession |
| `extract_heteroatoms` | Ligands + SMILES from a protein's PDB structures |
| `analyze_similarity` | Rank ligand records vs a target SMILES (Tanimoto) |
| `run_pipeline` | Extract + rank in one call (preferred) |
| `compare_smiles` | Tanimoto similarity between two molecules |

## Run the MCP server on its own

```bat
python -m mcp_server.server
```

It speaks JSON-RPC over stdio, so you normally launch it from an MCP client
rather than by hand. To use it in **Claude Desktop**, copy
`claude_desktop_config.example.json` into your Claude Desktop config
(`%APPDATA%\Claude\claude_desktop_config.json`), fix the `cwd` path, and restart
Claude Desktop. TrackMyPDB tools then appear in the chat.

## Run the in-app chat

```bat
pip install -r requirements.txt
set ANTHROPIC_API_KEY=sk-ant-...
streamlit run streamlit_app.py
```

Open the app, choose **AI Assistant** in the sidebar, and chat. (You can also
paste the key into the sidebar box instead of setting the env var.)

## Design notes

* `mcp_server/st_shim.py` lets the Streamlit-coupled backend run head-less in
  the MCP process by turning every `st.*` call into a no-op. This keeps stdout
  clean for the MCP protocol.
* Extraction hits public APIs (RCSB, PDBe, PubChem), so tools cap the number of
  PDB structures per UniProt ID (`max_pdbs_per_uniprot`, default 5).
* The model name is configurable via `ANTHROPIC_MODEL`; set it to whatever your
  account supports.
