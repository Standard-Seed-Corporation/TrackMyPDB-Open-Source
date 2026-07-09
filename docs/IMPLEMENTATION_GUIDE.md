# TrackMyPDB Agentic Layer Implementation Guide

## 🎯 Overview

This implementation adds an **intelligent agentic layer** to TrackMyPDB using:

- **Claude API** - Advanced language model for reasoning and conversation
- **MCP Server** - Model Context Protocol for tool exposure
- **Streamlit Chat UI** - Modern web interface for user interaction
- **Agent Framework** - Orchestration of complex bioinformatics workflows

## 📦 What's Included

### Core Components

1. **`mcp_trackmypdb_server.py`** - MCP Server
   - Exposes TrackMyPDB functions as tools
   - Heteroatom extraction tools
   - Molecular similarity analysis tools
   - Structure validation and analysis

2. **`trackmypdb_agent.py`** - Intelligent Agent
   - Claude-powered research assistant
   - Multi-turn conversation management
   - Tool orchestration
   - Research context tracking
   - Session management

3. **`chat_interface.py`** - Web UI
   - Streamlit-based chat interface
   - Real-time conversation display
   - Research context sidebar
   - Session management (save/load)
   - Quick template workflows

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8+
- Anthropic API key (get from https://console.anthropic.com/)
- TrackMyPDB repository (already included)

### Step 1: Install Dependencies

```bash
# Install required packages
pip install anthropic streamlit pandas numpy requests rdkit

# Or use the existing requirements
cd TrackMyPDB-Open-Source-sakeer
pip install -r requirements.txt

# Add Anthropic SDK
pip install anthropic
```

### Step 2: Configure API Key

**Option A: Environment Variable (Recommended)**
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

**Option B: In Application**
- Set via the chat interface sidebar
- Or pass to `TrackMyPDBAgent()` directly

### Step 3: Verify Installation

```bash
python trackmypdb_agent.py
```

You should see:
```
Available tools: 5
  - extract_heteroatoms
  - analyze_molecular_similarity
  - get_pdb_structures
  - validate_smiles
  - batch_heteroatom_extraction
```

## 🎮 Using the Chat Interface

### Launch the App

```bash
streamlit run chat_interface.py
```

The app will open at `http://localhost:8501`

### Features

#### 💬 Chat Area
- Ask questions naturally about protein structures and ligands
- Get intelligent responses powered by Claude
- Multi-turn conversations with context memory

#### ⚙️ Configuration Sidebar
- **API Key Setup** - Enter your Anthropic API key
- **Research Context** - Track proteins, ligands, and analyses
- **Session Management** - Save/load chat sessions
- **Quick Templates** - Pre-built workflow templates

#### 📝 Quick Templates

1. **Protein Analysis**
   - Extract heteroatoms from UniProt proteins
   - Analyze PDB structures
   - Identify ligands and cofactors

2. **Ligand Similarity**
   - Find compounds similar to target molecule
   - Compare structures using Morgan fingerprints
   - Identify binding preferences

3. **Structure Extraction**
   - Get all PDB structures for a protein
   - Extract heteroatoms and ligands
   - Analyze structural details

4. **Batch Analysis**
   - Process multiple proteins simultaneously
   - Generate comprehensive reports
   - Export results

## 📝 Example Workflows

### Example 1: Simple Protein Analysis

**User Input:**
```
Analyze the UniProt protein Q9UNQ0 and extract heteroatoms from its PDB structures
```

**Agent Response:**
The agent will:
1. Get PDB structures for Q9UNQ0
2. Extract heteroatoms from those structures
3. Provide information about ligands and cofactors
4. Suggest related analyses

### Example 2: Ligand Discovery

**User Input:**
```
I'm interested in molecules similar to aspirin (CC(=O)Oc1ccccc1C(=O)O). 
Can you find similar compounds in protein P37231's structures?
```

**Agent Response:**
The agent will:
1. Validate the SMILES string
2. Extract heteroatoms from P37231
3. Compare each ligand to aspirin
4. Return ranked similarity results
5. Explain structural similarities

### Example 3: Complex Research Workflow

**User Input:**
```
I'm researching kinase inhibitors. 
Here are proteins of interest: P37231, P06276, Q9UNQ0

1. Extract heteroatoms from all proteins
2. Find molecules similar to known kinase inhibitors
3. Identify common binding preferences
4. Suggest promising lead compounds
```

**Agent Response:**
The agent will:
1. Process all proteins in batch
2. Perform comparative analysis
3. Generate comprehensive findings
4. Provide actionable recommendations

## 🛠️ API Reference

### TrackMyPDBAgent Class

```python
from trackmypdb_agent import TrackMyPDBAgent

# Initialize
agent = TrackMyPDBAgent(
    api_key="sk-...",  # Optional, uses env var if not provided
    model="claude-opus-4-6"  # Latest Claude model
)

# Chat
response = agent.chat("What heteroatoms are in Q9UNQ0?")

# Get history
history = agent.get_history()

# Save/Load conversations
agent.save_conversation("my_research.json")
agent.load_conversation("my_research.json")

# Clear history
agent.clear_history()
```

### MCP Server Tools

#### 1. Extract Heteroatoms
```json
{
  "name": "extract_heteroatoms",
  "input": {
    "uniprot_ids": ["Q9UNQ0", "P37231"],
    "max_results": 1000
  }
}
```

#### 2. Analyze Molecular Similarity
```json
{
  "name": "analyze_molecular_similarity",
  "input": {
    "target_smiles": "CCO",
    "molecule_list": ["CC(=O)O", "c1ccccc1"],
    "morgan_radius": 2,
    "fingerprint_bits": 2048,
    "min_similarity": 0.2,
    "top_n": 50
  }
}
```

#### 3. Get PDB Structures
```json
{
  "name": "get_pdb_structures",
  "input": {
    "uniprot_id": "Q9UNQ0"
  }
}
```

#### 4. Validate SMILES
```json
{
  "name": "validate_smiles",
  "input": {
    "smiles": "CCO"
  }
}
```

#### 5. Batch Extraction
```json
{
  "name": "batch_heteroatom_extraction",
  "input": {
    "uniprot_ids": ["Q9UNQ0", "P37231"],
    "output_format": "json"
  }
}
```

## 🔧 Configuration

### Environment Variables

```bash
# Required
export ANTHROPIC_API_KEY='sk-...'

# Optional
export CLAUDE_MODEL='claude-opus-4-6'  # Default model
export MCP_SERVER_PORT=8000  # For standalone MCP server
```

### Model Selection

The agent supports all Claude models:
- `claude-opus-4-6` - Latest, most capable (recommended)
- `claude-sonnet-4-6` - Faster, good for real-time
- `claude-haiku-4-5` - Fastest, for simple tasks

## 📊 Output & Export

### Save Research Sessions

The chat interface allows you to:
1. **Save conversations** - `💾 Save Chat` button
2. **Export research context** - View as JSON
3. **Download results** - Session data with full chat history

### Output Formats

Sessions save as JSON:
```json
{
  "timestamp": "2025-01-15T10:30:00",
  "messages": [
    {
      "role": "user",
      "content": "...",
      "timestamp": "..."
    },
    {
      "role": "assistant",
      "content": "...",
      "timestamp": "..."
    }
  ],
  "research_context": {
    "proteins": ["Q9UNQ0"],
    "ligands": ["aspirin", "ibuprofen"],
    "analyses_performed": [...]
  }
}
```

## 🚀 Advanced Usage

### Custom Workflows

```python
from trackmypdb_agent import TrackMyPDBAgent, create_protein_analysis_workflow

agent = TrackMyPDBAgent()

# Use workflow helper
workflow = create_protein_analysis_workflow(["Q9UNQ0", "P37231"])
response = agent.chat(workflow)
```

### Batch Processing

```python
# Process multiple proteins
proteins = ["Q9UNQ0", "P37231", "P06276"]
response = agent.chat(f"Extract heteroatoms from: {', '.join(proteins)}")
```

### Research Sessions

```python
from trackmypdb_agent import AgentSession

# Create session
session = AgentSession("my_research_2025-01-15")

# Add context
session.set_research_context({
    "project": "Kinase Inhibitor Discovery",
    "target_family": "Serine/Threonine Kinases"
})

# Chat within session
response = session.chat("Find inhibitor-like molecules...")

# Get session info
print(session.get_session_info())
```

## 🐛 Troubleshooting

### Issue: "API Key not set"
```python
# Solution 1: Set environment variable
export ANTHROPIC_API_KEY='your-key'

# Solution 2: Pass to agent
agent = TrackMyPDBAgent(api_key='your-key')

# Solution 3: Set in Streamlit sidebar
```

### Issue: "Module not found: backend"
```bash
# Ensure you're in the correct directory
cd /home/claude

# Or add to path in Python
import sys
sys.path.insert(0, '/path/to/TrackMyPDB-Open-Source-sakeer')
```

### Issue: "RDKit not available"
```bash
# Install with conda (recommended)
conda install -c conda-forge rdkit

# Or via pip
pip install rdkit-pypi
```

### Issue: "Request timeout"
- Check internet connection
- Verify API keys have quota
- Retry with shorter input

## 📚 Documentation

### Key Files

- **`mcp_trackmypdb_server.py`** - Server implementation
  - Define tools for Claude
  - Execute heteroatom operations
  - Manage similarity analysis

- **`trackmypdb_agent.py`** - Agent implementation
  - Claude API integration
  - Multi-turn conversations
  - History management

- **`chat_interface.py`** - UI implementation
  - Streamlit app
  - Session management
  - Real-time chat

## 🔐 Security Best Practices

1. **Never commit API keys** - Use environment variables
2. **Use `.env` files** - For local development
3. **Validate inputs** - Especially SMILES strings
4. **Handle errors gracefully** - Catch exceptions
5. **Rate limiting** - Respect API quotas

## 📈 Performance Tips

1. **Batch operations** - Process multiple proteins at once
2. **Cache results** - Reuse previous extractions
3. **Use similarity threshold** - Filter low-similarity results
4. **Optimize fingerprints** - Balance accuracy vs speed
5. **Monitor API usage** - Track token consumption

## 🤝 Contributing

To extend the agent:

1. **Add new tools** - Modify `mcp_trackmypdb_server.py`
2. **Enhance prompts** - Update `_create_system_prompt()` in agent
3. **Add UI features** - Extend `chat_interface.py`
4. **Integrate services** - Add MCP servers for other databases

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review example workflows
3. Test with simple inputs first
4. Check API key validity
5. Review logs for error messages

## 📄 License

This implementation is licensed under MIT License, same as TrackMyPDB.

## 🎓 Educational Resources

- [MCP Documentation](https://modelcontextprotocol.io/)
- [Claude API Docs](https://docs.anthropic.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [RDKit Documentation](https://www.rdkit.org/docs/)

---

**Happy researching with TrackMyPDB! 🧬🔍**
