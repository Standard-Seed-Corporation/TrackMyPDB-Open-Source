"""
TrackMyPDB Diagnostic Script
Run this to find exactly what's failing.
Usage: python diagnose.py
"""
import sys
import os

print("=" * 60)
print("TrackMyPDB Diagnostic")
print("=" * 60)

print(f"\nPython version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Check required packages
print("\n--- Checking Dependencies ---")
packages = ["tqdm", "rdkit", "pandas", "requests", "numpy", "streamlit", "openai"]
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  [OK]   {pkg}")
    except ImportError:
        print(f"  [MISS] {pkg}  <-- NOT INSTALLED")
        missing.append(pkg)

# Check backend folder
print("\n--- Checking Backend Folder ---")
here = os.getcwd()
candidates = [
    os.path.join(here, "backend"),
    os.path.join(here, "..", "backend"),
]
backend_found = None
for c in candidates:
    if os.path.isdir(c):
        print(f"  [OK]   Found backend at: {os.path.abspath(c)}")
        backend_found = os.path.dirname(os.path.abspath(c))
        break
    else:
        print(f"  [--]   Not at: {os.path.abspath(c)}")

# Try importing backend
print("\n--- Trying to Import Backend ---")
if backend_found:
    sys.path.insert(0, backend_found)
    try:
        from backend.heteroatom_extractor import HeteroatomExtractor
        print("  [OK]   HeteroatomExtractor imported successfully!")
        # Try a live call
        print("\n--- Testing Live PDB Fetch ---")
        ext = HeteroatomExtractor()
        pdbs = ext.get_pdbs_for_uniprot("Q9UNQ0")
        print(f"  [OK]   Got {len(pdbs)} PDB structures: {pdbs[:5]}")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
else:
    print("  [FAIL] backend folder not found")

print("\n" + "=" * 60)
if missing:
    print(f"ACTION NEEDED: Install missing packages:")
    print(f"  pip install {' '.join(missing)}")
else:
    print("All dependencies present!")
print("=" * 60)
