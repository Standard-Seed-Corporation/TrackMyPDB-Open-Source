# TrackMyPDB Agentic Layer - Quick Start Guide

## ⚡ 5-Minute Setup

### 1. Get Your API Key
- Go to https://console.anthropic.com/
- Create an account or login
- Copy your API key

### 2. Install Requirements
```bash
pip install -r requirements_agent.txt
```

### 3. Set API Key
```bash
export ANTHROPIC_API_KEY='sk-your-key-here'
```

### 4. Launch Chat Interface
```bash
streamlit run chat_interface.py
```

### 5. Start Chatting!
Open http://localhost:8501 and start asking questions about proteins and ligands.

---

## 🎯 Quick Examples

### Example 1: Analyze a Protein (30 seconds)

**In the chat box, type:**
```
Extract heteroatoms from UniProt protein Q9UNQ0
```

**The agent will:**
- Find all PDB structures for this protein
- Extract heteroatoms (ligands, cofactors)
- Show you what was found

### Example 2: Find Similar Molecules (2 minutes)

**In the chat box, type:**
```
I have a molecule with SMILES: CCO (ethanol)
Find similar compounds in protein P37231
```

**The agent will:**
- Get ligands from the protein
- Compare each to your molecule
- Show most similar ones ranked by similarity score

### Example 3: Complete Research Workflow (5 minutes)

**In the chat box, type:**
```
I'm researching inhibitors for kinase targets.
Please:
1. Extract heteroatoms from proteins: Q9UNQ0, P37231, P06276
2. Identify any known kinase inhibitors
3. Find molecules similar to aspirin (CC(=O)Oc1ccccc1C(=O)O)
4. Rank them by similarity and suggest promising leads
```

**The agent will:**
- Process all proteins
- Find similar compounds
- Provide ranked recommendations
- Explain why they're promising

---

## 📋 Template-Based Workflows

Don't like typing? Use templates in the sidebar!

1. **Protein Analysis** - Extract heteroatoms from proteins
   - Paste UniProt IDs: `Q9UNQ0, P37231, P06276`
   - Click "Analyze"

2. **Ligand Similarity** - Find similar compounds
   - Paste target SMILES: `CCO`
   - Paste protein IDs: `Q9UNQ0, P37231`
   - Click "Analyze"

3. **Structure Extraction** - Get all PDB structures
   - Enter protein ID: `Q9UNQ0`
   - Click "Extract"

4. **Batch Analysis** - Process many proteins
   - Paste IDs (one per line)
   - Click "Process Batch"

---

## 🎓 Common Questions

### Q: What are UniProt IDs?
**A:** Unique identifiers for proteins. Example: `Q9UNQ0` is a valid ID.
- Find them at: https://www.uniprot.org/

### Q: What are SMILES?
**A:** Text representation of molecules. Examples:
- `CCO` = Ethanol
- `CC(=O)O` = Acetic acid
- `CC(=O)Oc1ccccc1C(=O)O` = Aspirin

### Q: How long does analysis take?
**A:** Usually 30 seconds to 2 minutes depending on:
- Number of proteins
- Number of heteroatoms
- Network speed

### Q: Can I save my research?
**A:** Yes! Use the `💾 Save Chat` button in the sidebar.
- Saves entire conversation
- Includes all research context
- Can be loaded later

---

## 🚀 Next Steps

### Learn More
- Read `IMPLEMENTATION_GUIDE.md` for detailed docs
- Check example workflows in the sidebar
- Explore the agent's capabilities

### Advanced Usage
```python
# Use programmatically
from trackmypdb_agent import TrackMyPDBAgent

agent = TrackMyPDBAgent()
response = agent.chat("Analyze protein Q9UNQ0")
print(response)
```

### Integrate with Your Research
- Save conversations for reproducibility
- Export results as JSON
- Use in automation pipelines
- Build custom workflows

---

## ⚠️ Troubleshooting

### "API Key Error"
```bash
# Check if key is set
echo $ANTHROPIC_API_KEY

# If empty, set it
export ANTHROPIC_API_KEY='sk-your-key'
```

### "Module not found"
```bash
# Reinstall dependencies
pip install --upgrade -r requirements_agent.txt
```

### "Connection timeout"
- Check internet connection
- Verify API key is valid
- Retry with shorter requests
- Check API status: https://status.anthropic.com

---

## 📚 Useful Links

- **Anthropic Console:** https://console.anthropic.com/
- **UniProt Database:** https://www.uniprot.org/
- **PDB Database:** https://www.rcsb.org/
- **SMILES Generator:** https://www.chemspider.com/
- **Documentation:** See `IMPLEMENTATION_GUIDE.md`

---

## 💡 Tips & Tricks

1. **Be specific** - More details = better responses
   - ❌ "Analyze this protein"
   - ✅ "Extract heteroatoms from protein Q9UNQ0 and identify kinase inhibitors"

2. **Use context** - Refer back to previous findings
   - "Based on the heteroatoms we found, what's similar to aspirin?"

3. **Ask for explanations** - Get insights
   - "Why are these compounds similar?"
   - "What does this tell us about the binding site?"

4. **Save often** - Don't lose your work
   - Use `💾 Save Chat` regularly

5. **Batch operations** - More efficient than one-by-one
   - Instead of asking about 5 proteins separately, ask about all 5

---

## 🎉 You're Ready!

Now go explore the world of bioinformatics with TrackMyPDB and Claude!

**Happy researching! 🧬🔍**

---

*Need help? Check the full documentation in `IMPLEMENTATION_GUIDE.md`*
