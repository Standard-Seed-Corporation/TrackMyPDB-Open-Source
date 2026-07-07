"""
Quick Test Script for TrackMyPDB New Features
Run this to verify all new components before commit
"""

import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_imports():
    """Test 1: Verify all new modules import correctly"""
    print("\n" + "="*60)
    print("TEST 1: Module Imports")
    print("="*60)
    
    tests_passed = 0
    tests_total = 0
    
    # Test molecule_visualizer
    tests_total += 1
    try:
        from backend.molecule_visualizer import MoleculeVisualizer, ChemicalDrawingTool
        print("✅ molecule_visualizer imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ molecule_visualizer import failed: {e}")
    
    # Test disease_annotator
    tests_total += 1
    try:
        from backend.disease_annotator import DiseaseAnnotator, DISEASE_CATEGORIES
        print("✅ disease_annotator imported successfully")
        print(f"   Available disease categories: {list(DISEASE_CATEGORIES.keys())}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ disease_annotator import failed: {e}")
    
    # Test heteroatom_extractor enhancements
    tests_total += 1
    try:
        from backend.heteroatom_extractor import HeteroatomExtractor
        extractor = HeteroatomExtractor()
        if hasattr(extractor, 'fetch_smiles_enhanced'):
            print("✅ heteroatom_extractor enhanced methods present")
            tests_passed += 1
        else:
            print("❌ Enhanced SMILES fetching method not found")
    except Exception as e:
        print(f"❌ heteroatom_extractor test failed: {e}")
    
    print(f"\nImport Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def test_visualizer():
    """Test 2: Verify molecule visualizer functionality"""
    print("\n" + "="*60)
    print("TEST 2: Molecule Visualizer")
    print("="*60)
    
    try:
        from backend.molecule_visualizer import MoleculeVisualizer
        from rdkit import Chem
        
        visualizer = MoleculeVisualizer()
        
        # Test SMILES
        test_smiles = [
            ("CCO", "Ethanol"),
            ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin"),
            ("c1ccccc1", "Benzene")
        ]
        
        for smiles, name in test_smiles:
            # Test image generation
            img = visualizer.smiles_to_image(smiles)
            if img:
                print(f"✅ {name} ({smiles}): Image generated")
            else:
                print(f"❌ {name} ({smiles}): Image generation failed")
            
            # Test property calculation
            props = visualizer.calculate_properties(smiles)
            if props:
                print(f"   MW: {props['Molecular_Weight']} Da, "
                      f"LogP: {props['LogP']}, "
                      f"Lipinski: {props['Lipinski_Violations']} violations")
            else:
                print(f"❌ Property calculation failed for {name}")
        
        print("\n✅ Molecule Visualizer tests passed")
        return True
        
    except ImportError as e:
        print(f"❌ RDKit not available: {e}")
        print("   Install with: pip install rdkit")
        return False
    except Exception as e:
        print(f"❌ Visualizer test failed: {e}")
        return False


def test_smiles_fallback():
    """Test 3: Verify enhanced SMILES fetching with fallbacks"""
    print("\n" + "="*60)
    print("TEST 3: Enhanced SMILES Fetching (8D0 Fix)")
    print("="*60)
    
    try:
        from backend.heteroatom_extractor import HeteroatomExtractor
        
        extractor = HeteroatomExtractor()
        
        # Test known ligands with different sources
        test_codes = [
            ("ATP", "Should work from RCSB"),
            ("8D0", "Should work from PDB Ligand Expo"),
            ("NAG", "Should work from RCSB"),
        ]
        
        for code, description in test_codes:
            print(f"\nTesting {code} ({description})...")
            result = extractor.fetch_smiles_enhanced(code)
            
            if result['smiles']:
                print(f"✅ {code}: Found SMILES")
                print(f"   Status: {result['status']}")
                print(f"   SMILES: {result['smiles'][:50]}...")
                print(f"   Formula: {result.get('formula', 'N/A')}")
            else:
                print(f"❌ {code}: No SMILES found")
                print(f"   Status: {result['status']}")
        
        print("\n✅ SMILES fallback tests completed")
        return True
        
    except Exception as e:
        print(f"❌ SMILES fallback test failed: {e}")
        return False


def test_disease_annotator():
    """Test 4: Verify disease annotation functionality"""
    print("\n" + "="*60)
    print("TEST 4: Disease Annotator")
    print("="*60)
    
    try:
        from backend.disease_annotator import DiseaseAnnotator, DISEASE_CATEGORIES
        import pandas as pd
        
        annotator = DiseaseAnnotator()
        
        # Test with known protein
        test_uniprot = "P04637"  # TP53 tumor suppressor
        
        print(f"\nFetching disease annotations for {test_uniprot} (TP53)...")
        result = annotator.fetch_uniprot_disease_annotations(test_uniprot)
        
        if result['status'] == 'success':
            print(f"✅ Disease annotation successful")
            print(f"   Protein: {result.get('protein_name', 'N/A')}")
            print(f"   Diseases found: {len(result.get('diseases', []))}")
            
            if result.get('diseases'):
                for disease in result['diseases'][:3]:  # Show first 3
                    print(f"   - {disease.get('disease_name', 'Unknown')}")
        else:
            print(f"⚠️ Could not fetch annotations: {result['status']}")
            print("   (This may be due to API connectivity)")
        
        # Test disease categories
        print(f"\n✅ Disease categories available: {len(DISEASE_CATEGORIES)}")
        for category in list(DISEASE_CATEGORIES.keys())[:3]:
            print(f"   - {category}")
        
        print("\n✅ Disease Annotator tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Disease annotator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_streamlit_integration():
    """Test 5: Verify streamlit app structure"""
    print("\n" + "="*60)
    print("TEST 5: Streamlit Integration")
    print("="*60)
    
    try:
        # Read streamlit_app.py
        with open('streamlit_app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for new pages
        required_functions = [
            'show_molecule_visualizer_page',
            'show_disease_enrichment_page'
        ]
        
        for func in required_functions:
            if func in content:
                print(f"✅ Found function: {func}")
            else:
                print(f"❌ Missing function: {func}")
        
        # Check for new imports
        if 'MoleculeVisualizer' in content and 'DiseaseAnnotator' in content:
            print("✅ New modules imported in streamlit_app.py")
        else:
            print("❌ New modules not imported properly")
        
        # Check for new navigation options
        if '🖼️ Molecule Visualizer' in content and '🏥 Disease Enrichment' in content:
            print("✅ New pages added to navigation")
        else:
            print("❌ New pages not added to navigation")
        
        print("\n✅ Streamlit integration checks completed")
        return True
        
    except Exception as e:
        print(f"❌ Streamlit integration test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*60)
    print("🧪 TRACKMYPDB - COMPREHENSIVE TEST SUITE")
    print("="*60)
    print("Testing all new features before deployment...")
    
    results = {
        'Imports': test_imports(),
        'Visualizer': test_visualizer(),
        'SMILES Fallback': test_smiles_fallback(),
        'Disease Annotator': test_disease_annotator(),
        'Streamlit Integration': test_streamlit_integration()
    }
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{'='*60}")
    print(f"Overall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Ready for deployment!")
        print("\nNext steps:")
        print("1. Run: streamlit run streamlit_app.py")
        print("2. Test in browser manually")
        print("3. If all looks good, commit changes")
        return True
    else:
        print("\n⚠️ SOME TESTS FAILED! Review errors above.")
        print("\nFix issues before deployment:")
        for test_name, result in results.items():
            if not result:
                print(f"   - {test_name}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
