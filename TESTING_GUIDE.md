# 🧪 TrackMyPDB - Local Testing Guide
## Complete Testing Protocol Before Deployment

**Created:** 2026-07-07  
**Author:** Standard Seed Corporation  
**Purpose:** Ensure all 4 new features work correctly before git commit

---

## 📋 **Testing Checklist Overview**

- [ ] **Test 1:** Backend imports and dependencies
- [ ] **Test 2:** Molecule Visualizer page
- [ ] **Test 3:** Physicochemical properties calculation
- [ ] **Test 4:** PDB Ligand Expo fallback (8D0 issue fix)
- [ ] **Test 5:** Disease Enrichment page
- [ ] **Test 6:** Integration tests (end-to-end workflows)
- [ ] **Test 7:** Error handling and edge cases
- [ ] **Test 8:** Cross-page navigation

---

## 🚀 **Pre-Testing Setup**

### Step 1: Verify File Structure
```bash
# Run this in your project root directory
cd d:\ssc\TrackMyPDB-Open-Source

# Check all new files exist
dir backend\molecule_visualizer.py
dir backend\disease_annotator.py

# Should see both files listed
```

### Step 2: Verify Python Environment
```powershell
# Check Python version (should be 3.7+)
python --version

# Verify RDKit is installed
python -c "from rdkit import Chem; print('RDKit OK')"

# If RDKit not installed:
pip install rdkit
```

### Step 3: Check Dependencies
```powershell
# Verify all required packages
pip install -r requirements.txt

# Specifically check for new dependencies
python -c "import streamlit, pandas, requests; print('Core packages OK')"
python -c "from rdkit import Chem, Descriptors; print('RDKit OK')"
```

---

## 🧪 **TEST 1: Backend Import Verification**

### Purpose
Ensure all new modules load without errors

### Steps
```powershell
# Test 1.1: Import molecule_visualizer
python -c "from backend.molecule_visualizer import MoleculeVisualizer, ChemicalDrawingTool; print('✅ Molecule visualizer imports OK')"

# Test 1.2: Import disease_annotator
python -c "from backend.disease_annotator import DiseaseAnnotator, DISEASE_CATEGORIES; print('✅ Disease annotator imports OK')"

# Test 1.3: Import updated heteroatom_extractor
python -c "from backend.heteroatom_extractor import HeteroatomExtractor; print('✅ Heteroatom extractor imports OK')"

# Test 1.4: Test new methods exist
python -c "from backend.heteroatom_extractor import HeteroatomExtractor; e = HeteroatomExtractor(); print('✅ Enhanced SMILES fetching:', hasattr(e, 'fetch_smiles_enhanced'))"
```

### Expected Output
```
✅ Molecule visualizer imports OK
✅ Disease annotator imports OK
✅ Heteroatom extractor imports OK
✅ Enhanced SMILES fetching: True
```

### If Failed
- Check for typos in file paths
- Verify Python path includes backend directory
- Check for syntax errors: `python -m py_compile backend/molecule_visualizer.py`

---

## 🧪 **TEST 2: Streamlit Application Launch**

### Purpose
Verify the app starts without errors

### Steps
```powershell
# Start the application
streamlit run streamlit_app.py
```

### Expected Output
- Browser opens to `http://localhost:8501`
- No red error messages in terminal
- Home page displays correctly
- Sidebar shows 7 pages (including new ones)

### Checklist
- [ ] App launches without import errors
- [ ] Sidebar shows "🖼️ Molecule Visualizer"
- [ ] Sidebar shows "🏥 Disease Enrichment"
- [ ] No warnings about missing modules (except optional ones)

### If Failed
Check terminal output for:
```
ImportError: No module named 'backend.molecule_visualizer'
  → File not created or wrong path

SyntaxError: invalid syntax
  → Check code for syntax errors

ModuleNotFoundError: No module named 'rdkit'
  → Install RDKit: pip install rdkit
```

---

## 🧪 **TEST 3: Molecule Visualizer Page**

### Purpose
Test 2D visualization and property calculation

### Steps

#### Test 3.1: Basic SMILES Input
1. Navigate to "🖼️ Molecule Visualizer"
2. Go to "✍️ Manual SMILES" tab
3. Enter: `CCO` (ethanol)
4. Click outside text area

**Expected Results:**
- ✅ Green message: "Valid SMILES structure"
- ✅ 2D molecular structure displays on left
- ✅ Properties table displays on right
- ✅ Molecular Weight: ~46 Da
- ✅ Formula: C2H6O

#### Test 3.2: Complex Molecule (Aspirin)
1. Enter: `CC(=O)Oc1ccccc1C(=O)O`

**Expected Results:**
- ✅ Structure displays correctly (aromatic ring visible)
- ✅ Molecular Weight: ~180 Da
- ✅ Lipinski violations: 0
- ✅ QED score displayed

#### Test 3.3: Name Resolution
1. Go to "🔍 Search by Name" tab
2. Enter: `aspirin`
3. Click "🔍 Search"

**Expected Results:**
- ✅ Success message with SMILES
- ✅ SMILES returned: `CC(=O)Oc1ccccc1C(=O)O`
- ✅ Structure auto-displays

#### Test 3.4: Invalid SMILES
1. Go to "✍️ Manual SMILES" tab
2. Enter: `INVALID123`

**Expected Results:**
- ❌ Red message: "Invalid SMILES - please check syntax"
- No structure displayed

#### Test 3.5: File Upload (Optional)
1. Go to "🖼️ Upload MOL/SDF" tab
2. Create a test MOL file (or skip if not available)

#### Test 3.6: Download Functions
1. Enter valid SMILES: `CCO`
2. Click "📥 Download Structure (PNG)"
3. Click "📥 Download Properties (CSV)"

**Expected Results:**
- ✅ PNG file downloads (~50-200 KB)
- ✅ CSV file downloads with all properties
- ✅ Files open correctly

### Property Verification Table

| SMILES | MW (Da) | LogP | HBD | HBA | Violations | QED |
|--------|---------|------|-----|-----|------------|-----|
| CCO | 46 | ~-0.3 | 1 | 1 | 0 | ~0.7 |
| CC(=O)Oc1ccccc1C(=O)O | 180 | ~1.2 | 1 | 4 | 0 | ~0.6 |
| c1ccccc1 | 78 | ~1.7 | 0 | 0 | 0 | ~0.4 |

### If Failed
- **No structure displays:** Check RDKit installation
- **Properties empty:** Verify Descriptors module imported
- **Name resolution fails:** Check internet connection (PubChem API)

---

## 🧪 **TEST 4: PDB Ligand Expo Fallback (8D0 Fix)**

### Purpose
Verify the enhanced SMILES fetching resolves the 8D0 issue

### Critical Test Case

#### Test 4.1: Extract from 5XRA (Should Find 8D0)
1. Navigate to "🔍 Heteroatom Extraction"
2. Enter UniProt ID: `P00533` (EGFR - has 8D0 ligand)
3. Click "🚀 Start Heteroatom Extraction"

**Monitor Terminal Output:**
Look for these messages:
```
Processing X heteroatoms from XXXX: ..., 8D0, ...
RCSB failed for 8D0, trying PDB Ligand Expo...
✅ Found 8D0 in PDB Ligand Expo!
```

**Expected Results:**
- ✅ 8D0 appears in heteroatom list
- ✅ 8D0 has valid SMILES (not empty)
- ✅ Status: `pdb_ligand_expo_sdf` or `pdb_ligand_expo_model`
- ✅ Chemical formula populated

#### Test 4.2: Verify in Results Table
**Check the results DataFrame:**
- Find row with Heteroatom_Code = "8D0"
- SMILES column should NOT be empty
- Status should indicate PDB Ligand Expo source

#### Test 4.3: Fallback Order Test
Create a test script to verify fallback logic:

```python
# test_fallback.py
from backend.heteroatom_extractor import HeteroatomExtractor

extractor = HeteroatomExtractor()

# Test known ligands
test_codes = ['ATP', '8D0', 'NAG', 'HEM']

for code in test_codes:
    result = extractor.fetch_smiles_enhanced(code)
    print(f"{code}: {result['status']} | SMILES: {result['smiles'][:50]}...")
```

Run:
```powershell
python test_fallback.py
```

**Expected Output:**
```
ATP: success | SMILES: c1nc(c2c(n1)n(cn2)C3C(C(C(O3)COP(=O)(O)OP...
8D0: pdb_ligand_expo_sdf | SMILES: [Valid SMILES string]...
NAG: success | SMILES: CC(=O)NC1C(C(C(OC1O)CO)O)O...
HEM: pdb_ligand_expo_sdf | SMILES: [Valid SMILES string]...
```

### If Failed
- **8D0 still empty:** Check internet connection to files.wwpdb.org
- **RDKit import error:** Verify RDKit installed
- **Wrong status:** Check method execution order in fetch_smiles_enhanced

---

## 🧪 **TEST 5: Disease Enrichment Page**

### Purpose
Test disease annotation fetching and filtering

### Steps

#### Test 5.1: Manual UniProt Input
1. Navigate to "🏥 Disease Enrichment"
2. Select "Manual UniProt Input"
3. Enter:
   ```
   P04637
   P53_HUMAN
   ```
4. Click "🔍 Fetch Disease Annotations"

**Expected Results:**
- ✅ Creates dataset with 2 UniProt IDs
- ✅ Ready for enrichment

#### Test 5.2: Run Disease Enrichment
1. Click "🚀 Run Disease Enrichment"

**Monitor:**
- Progress bar appears
- Status updates: "Fetching disease info for P04637..."

**Expected Results:**
- ✅ Disease_Associations column added
- ✅ P04637 (TP53) shows cancer-related diseases
- ✅ Data displays in table

#### Test 5.3: Disease Category Filtering
1. Select categories: "Cancer", "Neurological"
2. Observe filtered results

**Expected Results:**
- ✅ Results filtered to matching diseases
- ✅ Count updates: "Filtered to X results"

#### Test 5.4: Custom Keywords
1. Enter custom keywords: `breast cancer, lung cancer`
2. Observe results

**Expected Results:**
- ✅ Only cancer-related entries shown
- ✅ Bar chart displays disease distribution

#### Test 5.5: Integration with Other Pages
1. First run a Similarity Analysis (any SMILES)
2. Fetch protein information
3. Go to Disease Enrichment
4. Select "Similarity Analysis Results"

**Expected Results:**
- ✅ Results load from session state
- ✅ UniProt_IDs column present
- ✅ Can enrich existing results

### Known UniProt IDs for Testing

| UniProt ID | Protein | Expected Diseases |
|------------|---------|-------------------|
| P04637 | TP53 | Cancer, Li-Fraumeni syndrome |
| P01308 | Insulin | Diabetes, hyperinsulinemia |
| P04062 | Glucokinase | Diabetes mellitus type 2 |
| P42574 | Caspase-3 | Cancer, neurodegeneration |

### If Failed
- **No diseases found:** Check UniProt API connectivity
- **"No UniProt_IDs column":** Need to fetch protein info first
- **Empty cache:** UniProt ID might be invalid

---

## 🧪 **TEST 6: Integration Tests (End-to-End)**

### Purpose
Test complete workflows across multiple features

### Workflow 1: Complete Drug Discovery Pipeline

1. **Extract heteroatoms:**
   - Input: `P00533` (EGFR)
   - Verify 8D0 extracted with SMILES

2. **Run similarity analysis:**
   - Target SMILES: Use 8D0's SMILES from extraction
   - Top N: 20
   - Min similarity: 0.3

3. **Visualize top hit:**
   - Copy SMILES of top result
   - Go to Molecule Visualizer
   - Check properties

4. **Enrich with disease data:**
   - Go to Disease Enrichment
   - Select similarity results
   - Run enrichment
   - Filter for "Cancer"

**Success Criteria:**
- ✅ All steps complete without errors
- ✅ Data flows between pages via session state
- ✅ CSV downloads work at each stage

### Workflow 2: SMILES Database Search → Disease Mapping

1. **Database search:**
   - Input SMILES: `CC(=O)Oc1ccccc1C(=O)O` (aspirin-like)
   - Search database
   - Get results

2. **Fetch protein info:**
   - Click "Fetch Protein Information"
   - Verify UniProt IDs populated

3. **Disease enrichment:**
   - Go to Disease Enrichment
   - Select "SMILES Database Search"
   - Run enrichment
   - Filter for "Cardiovascular"

**Success Criteria:**
- ✅ Aspirin-like compounds found
- ✅ COX proteins identified
- ✅ Cardiovascular diseases linked

---

## 🧪 **TEST 7: Error Handling & Edge Cases**

### Purpose
Verify graceful failure handling

### Test 7.1: Invalid Inputs
- [ ] Empty SMILES input → No error, just info message
- [ ] Invalid UniProt ID → Graceful failure message
- [ ] No internet connection → Clear error about API failure
- [ ] Empty results → "No results available" message

### Test 7.2: Session State Management
- [ ] Refresh page → Session state persists
- [ ] Switch pages and back → Data still there
- [ ] Clear button works → Session state cleared

### Test 7.3: Large Datasets
- [ ] 100+ heteroatoms → Progress bar works
- [ ] 50+ disease annotations → No timeout
- [ ] Large SMILES string → Truncated display

---

## 🧪 **TEST 8: Cross-Browser & Performance**

### Test 8.1: Browser Compatibility
Test in:
- [ ] Chrome/Edge (Chromium)
- [ ] Firefox
- [ ] Safari (if on Mac)

### Test 8.2: Performance Benchmarks
- [ ] Page load < 3 seconds
- [ ] SMILES visualization < 1 second
- [ ] Disease enrichment < 5 seconds per protein
- [ ] No memory leaks (check Task Manager)

---

## 📊 **Final Pre-Commit Checklist**

### Code Quality
- [ ] No syntax errors: `python -m py_compile backend/*.py`
- [ ] No unused imports
- [ ] Docstrings present for all functions
- [ ] Error messages user-friendly

### Functionality
- [ ] All 7 pages accessible
- [ ] All 4 new features working
- [ ] CSV downloads functional
- [ ] Session state managed correctly

### Data Integrity
- [ ] 8D0 ligand found with SMILES
- [ ] Properties calculated correctly
- [ ] Disease annotations accurate
- [ ] No data loss between pages

### Documentation
- [ ] README.md updated (if needed)
- [ ] Code comments clear
- [ ] Changelog entry added

---

## 🚨 **Common Issues & Solutions**

### Issue 1: "ModuleNotFoundError: No module named 'backend'"
**Solution:**
```powershell
# Verify path is set correctly in streamlit_app.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
```

### Issue 2: "RDKit not available"
**Solution:**
```powershell
pip uninstall rdkit
pip install rdkit
# Or use conda:
conda install -c conda-forge rdkit
```

### Issue 3: "8D0 SMILES still empty"
**Solution:**
- Test API directly:
```python
import requests
url = "https://files.wwpdb.org/pub/pdb/data/monomers/8/8D0/8D0_ideal.sdf"
print(requests.get(url).status_code)  # Should be 200
```

### Issue 4: "Session state not persisting"
**Solution:**
- Check Streamlit version: `streamlit --version` (should be >= 1.28)
- Verify session_state keys match exactly

### Issue 5: "UniProt API returns 404"
**Solution:**
- Verify UniProt ID format (P00000 or Q00000)
- Check API endpoint: `https://rest.uniprot.org/uniprotkb/P04637.json`

---

## 📝 **Testing Log Template**

Copy this to track your testing:

```markdown
## Testing Session: [DATE]

### Environment
- Python version: ____
- Streamlit version: ____
- RDKit installed: Yes/No
- OS: Windows/Mac/Linux

### Test Results

| Test ID | Feature | Status | Notes |
|---------|---------|--------|-------|
| TEST-1 | Backend imports | ✅/❌ | |
| TEST-2 | App launch | ✅/❌ | |
| TEST-3.1 | Basic SMILES viz | ✅/❌ | |
| TEST-3.2 | Complex molecule | ✅/❌ | |
| TEST-3.3 | Name resolution | ✅/❌ | |
| TEST-4.1 | 8D0 extraction | ✅/❌ | |
| TEST-5.1 | Manual UniProt | ✅/❌ | |
| TEST-5.2 | Disease enrichment | ✅/❌ | |
| TEST-6 | End-to-end workflow | ✅/❌ | |
| TEST-7 | Error handling | ✅/❌ | |

### Issues Found
1. [Description] → [Solution]
2. [Description] → [Solution]

### Ready for Commit? YES/NO
```

---

## 🎯 **Quick Test Command Sequence**

Run this complete test in PowerShell:

```powershell
# Quick smoke test
Write-Host "=== TrackMyPDB Testing ===" -ForegroundColor Green

# Test 1: Imports
Write-Host "`nTest 1: Checking imports..." -ForegroundColor Yellow
python -c "from backend.molecule_visualizer import MoleculeVisualizer; print('✅ Visualizer OK')"
python -c "from backend.disease_annotator import DiseaseAnnotator; print('✅ Annotator OK')"
python -c "from backend.heteroatom_extractor import HeteroatomExtractor; print('✅ Extractor OK')"

# Test 2: Launch app (will open browser)
Write-Host "`nTest 2: Launching app..." -ForegroundColor Yellow
Write-Host "Press Ctrl+C when done testing in browser" -ForegroundColor Cyan
streamlit run streamlit_app.py

# After manual testing in browser, you'll return here
Write-Host "`n=== Testing Complete ===" -ForegroundColor Green
```

---

## ✅ **Sign-Off**

When all tests pass:

```
✅ All imports successful
✅ App launches without errors
✅ Molecule Visualizer working
✅ Properties calculated correctly
✅ 8D0 ligand found with SMILES
✅ Disease enrichment functional
✅ Integration workflows complete
✅ Error handling graceful

APPROVED FOR COMMIT: [Your Name] [Date]
```

---

**Good luck with testing! 🚀**
