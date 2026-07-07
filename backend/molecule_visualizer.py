"""
TrackMyPDB Molecule Visualizer
@author: Anu Gamage, Standard Seed Corporation

Handles 2D molecular visualization and physicochemical property calculations
Licensed under MIT License - Open Source Project
"""

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Crippen, Lipinski, AllChem, rdMolDescriptors
from io import BytesIO
import base64
from PIL import Image
import pandas as pd
import requests


class MoleculeVisualizer:
    """Handle molecular visualization and property calculations"""
    
    def __init__(self):
        self.default_size = (400, 400)
    
    def smiles_to_image(self, smiles, size=(400, 400), highlight_atoms=None):
        """
        Convert SMILES to 2D molecular structure image
        
        Args:
            smiles (str): SMILES string
            size (tuple): Image size (width, height)
            highlight_atoms (list): Atom indices to highlight
            
        Returns:
            PIL Image or None
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Generate 2D coordinates
            AllChem.Compute2DCoords(mol)
            
            # Draw molecule
            if highlight_atoms:
                img = Draw.MolToImage(mol, size=size, highlightAtoms=highlight_atoms)
            else:
                img = Draw.MolToImage(mol, size=size)
            
            return img
        except Exception as e:
            st.error(f"Error generating image: {e}")
            return None
    
    def image_to_base64(self, img):
        """Convert PIL Image to base64 string for HTML embedding"""
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def display_molecule(self, smiles, caption="", size=(400, 400)):
        """Display molecule in Streamlit with caption"""
        img = self.smiles_to_image(smiles, size=size)
        if img:
            st.image(img, caption=caption, use_column_width=False)
            return True
        else:
            st.error(f"Invalid SMILES: {smiles}")
            return False
    
    def calculate_properties(self, smiles):
        """
        Calculate comprehensive physicochemical properties
        
        Returns:
            dict: Dictionary of molecular properties
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            properties = {
                # Basic properties
                'Molecular_Weight': round(Descriptors.MolWt(mol), 2),
                'Exact_Mass': round(Descriptors.ExactMolWt(mol), 2),
                'Formula': rdMolDescriptors.CalcMolFormula(mol),
                
                # Lipinski's Rule of Five
                'LogP': round(Crippen.MolLogP(mol), 2),
                'HBD': Lipinski.NumHDonors(mol),
                'HBA': Lipinski.NumHAcceptors(mol),
                
                # Topology
                'Rotatable_Bonds': Lipinski.NumRotatableBonds(mol),
                'Aromatic_Rings': Lipinski.NumAromaticRings(mol),
                'Aliphatic_Rings': Lipinski.NumAliphaticRings(mol),
                'Heavy_Atoms': Lipinski.HeavyAtomCount(mol),
                
                # Surface properties
                'TPSA': round(Descriptors.TPSA(mol), 2),
                'Molar_Refractivity': round(Crippen.MolMR(mol), 2),
                
                # Drug-likeness
                'Lipinski_Violations': self._lipinski_violations(mol),
                'QED': round(Descriptors.qed(mol), 3),
                
                # Complexity
                'Fraction_Csp3': round(Lipinski.FractionCsp3(mol), 3),
                'Num_Stereocenters': len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
            }
            
            return properties
            
        except Exception as e:
            st.error(f"Error calculating properties: {e}")
            return None
    
    def _lipinski_violations(self, mol):
        """Count Lipinski's Rule of Five violations"""
        violations = 0
        
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        if mw > 500:
            violations += 1
        if logp > 5:
            violations += 1
        if hbd > 5:
            violations += 1
        if hba > 10:
            violations += 1
            
        return violations
    
    def display_properties_table(self, properties):
        """Display properties in a formatted table"""
        if not properties:
            st.error("No properties to display")
            return
        
        # Create DataFrame for better display
        df = pd.DataFrame([properties]).T
        df.columns = ['Value']
        df.index.name = 'Property'
        
        st.dataframe(df, use_container_width=True)
        
        # Lipinski interpretation
        violations = properties['Lipinski_Violations']
        if violations == 0:
            st.success("✅ Passes Lipinski's Rule of Five - Good drug-likeness!")
        elif violations == 1:
            st.warning("⚠️ 1 Lipinski violation - Still potentially drug-like")
        else:
            st.error(f"❌ {violations} Lipinski violations - Poor drug-likeness")


class ChemicalDrawingTool:
    """Provide chemical structure input methods"""
    
    @staticmethod
    def simple_smiles_input():
        """Simple SMILES input with validation and name resolution"""
        st.subheader("🖊️ Input or Draw Structure")
        
        tab1, tab2, tab3 = st.tabs(["✍️ Manual SMILES", "🔍 Search by Name", "🖼️ Upload MOL/SDF"])
        
        with tab1:
            smiles = st.text_area(
                "Enter SMILES String",
                placeholder="Example: CCO (ethanol), c1ccccc1 (benzene), CC(=O)Oc1ccccc1C(=O)O (aspirin)",
                height=100,
                help="Enter valid SMILES representation"
            )
            
            if smiles:
                # Validate
                mol = Chem.MolFromSmiles(smiles.strip())
                if mol:
                    st.success("✅ Valid SMILES structure")
                    return smiles.strip()
                else:
                    st.error("❌ Invalid SMILES - please check syntax")
                    return None
        
        with tab2:
            compound_name = st.text_input("Chemical Name", placeholder="e.g., aspirin, caffeine, ATP")
            if st.button("🔍 Search") and compound_name:
                smiles = ChemicalDrawingTool.resolve_name_to_smiles(compound_name)
                if smiles:
                    st.success(f"✅ Found: `{smiles}`")
                    st.session_state['resolved_smiles'] = smiles
                    return smiles
                else:
                    st.error("❌ Could not resolve name to SMILES")
        
        with tab3:
            uploaded_file = st.file_uploader("Upload MOL/SDF file", type=['mol', 'sdf'])
            if uploaded_file:
                try:
                    content = uploaded_file.read().decode('utf-8')
                    mol = Chem.MolFromMolBlock(content)
                    if mol:
                        smiles = Chem.MolToSmiles(mol)
                        st.success(f"✅ Converted to SMILES: `{smiles}`")
                        return smiles
                    else:
                        st.error("❌ Could not parse molecular file")
                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")
        
        return None
    
    @staticmethod
    def resolve_name_to_smiles(compound_name):
        """Resolve chemical name to SMILES using PubChem"""
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/property/CanonicalSMILES/JSON"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
                return smiles
        except Exception as e:
            st.error(f"Error resolving name: {e}")
        
        return None
