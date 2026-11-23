# AI assistant and MCP server

TrackMyPDB can be driven in plain language. Two parts share one set of tools:

1. **MCP server** (`mcp_server/`): exposes the `trackmypdb` library as
   [Model Context Protocol](https://modelcontextprotocol.io) tools over stdio, for any
   MCP client (Claude Desktop, Cursor, ...).
2. **AI Assistant page** (`agent/`): a chat page in the web app. Claude answers by
   calling the same MCP server.

```
Streamlit "AI Assistant" page ──> agent/mcp_agent.py (Claude tool-use loop) ──┐
                                                                              │ stdio (MCP)
Claude Desktop / Cursor ──────────────────────────────────────────────────────┴─> mcp_server/server.py
                                                                                        │
                                                                                  trackmypdb library
```

## Tools

| Tool | What it does |
|------|--------------|
| `list_pdbs_for_uniprot` | PDB IDs for a UniProt accession, best first |
| `extract_heteroatoms` | Ligands + SMILES from a protein's PDB structures |
| `analyze_similarity` | Rank ligand records against a target SMILES (Tanimoto) |
| `run_pipeline` | Extract + rank in one call |
| `search_ligand_database` | Similarity search in the bundled offline ligand database |
| `compare_smiles` | Tanimoto similarity of two molecules |

Tool results are capped at 200 records to keep them within a model's context window.

## Use with Claude Desktop

Install with `pip install -e ".[agent]"`, then add this to Claude Desktop's
`claude_desktop_config.json` (on Windows: `%APPDATA%\Claude\claude_desktop_config.json`),
replacing the path with the absolute path of your clone:

```json
{
  "mcpServers": {
    "trackmypdb": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "env": { "PYTHONPATH": "C:\\path\\to\\TrackMyPDB-Open-Source" }
    }
  }
}
```

Restart Claude Desktop; the TrackMyPDB tools then appear in the chat. Use the full
path to the Python executable of your virtual environment as `command` if `python`
is not the right interpreter.

Check the connection without an API key or internet access:

```bash
python scripts/mcp_healthcheck.py
```

## In-app assistant

```bash
pip install -e ".[app,agent]"
streamlit run streamlit_app.py
```

Open **AI Assistant** and paste an Anthropic API key in the sidebar (it is kept only in
your browser session). Alternatively put it in a `.env` file (see `.env.example`).

**Shared keys on public deployments.** A key in the deployment's secrets or environment
is used for every visitor *only* if `TRACKMYPDB_ALLOW_SHARED_KEY=1` is also set. Leave
this off on public deployments, or every visitor can spend your credits.

Other settings: `ANTHROPIC_MODEL` (default `claude-sonnet-5-5`) and
`TRACKMYPDB_MAX_TOOL_ROUNDS` (default 8, the maximum rounds of tool calls per question).
