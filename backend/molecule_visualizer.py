"""
TrackMyPDB Molecule Visualizer
@author: Anu Gamage, Standard Seed Corporation

Handles 2D molecular visualization and physicochemical property calculations
Licensed under MIT License - Open Source Project
"""

# Force matplotlib to use non-GUI backend (fixes libXrender.so.1 error on Linux/WSL)
import matplotlib
matplotlib.use('Agg')

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
        Calculate core physicochemical properties (5 essential descriptors)
        
        Returns:
            dict: Dictionary of molecular properties (MW, LogP, HBD, HBA, TPSA)
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Calculate ONLY the 5 core properties using robust RDKit paths
            properties = {
                'Molecular_Weight': round(Descriptors.MolWt(mol), 2),
                'LogP': round(Descriptors.MolLogP(mol), 2),
                'HBD': Lipinski.NumHDonors(mol),
                'HBA': Lipinski.NumHAcceptors(mol),
                'TPSA': round(Descriptors.TPSA(mol), 2),
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
        """Simplified SMILES input - Manual entry or File upload only"""
        st.subheader("🖊️ Input Structure")
        
        # Only 2 tabs: Manual SMILES and File Upload
        tab1, tab2 = st.tabs(["✍️ Manual SMILES", "📁 File Upload"])
        
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
            st.markdown("Upload a file containing SMILES structures (.smi, .csv, .txt)")
            uploaded_file = st.file_uploader("Choose file", type=['smi', 'csv', 'txt', 'mol', 'sdf'])
            
            if uploaded_file:
                try:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_extension in ['mol', 'sdf']:
                        # Handle MOL/SDF files
                        content = uploaded_file.read().decode('utf-8')
                        mol = Chem.MolFromMolBlock(content)
                        if mol:
                            smiles = Chem.MolToSmiles(mol)
                            st.success(f"✅ Converted to SMILES: `{smiles}`")
                            return smiles
                        else:
                            st.error("❌ Could not parse molecular file")
                    else:
                        # Handle text-based files (SMILES)
                        content = uploaded_file.read().decode('utf-8')
                        # Try to extract first SMILES from file
                        lines = content.strip().split('\n')
                        if lines:
                            # Take first line or first column if CSV
                            first_line = lines[0]
                            if ',' in first_line:
                                smiles = first_line.split(',')[0].strip()
                            else:
                                smiles = first_line.strip()
                            
                            # Validate
                            mol = Chem.MolFromSmiles(smiles)
                            if mol:
                                st.success(f"✅ Loaded SMILES: `{smiles}`")
                                st.info(f"📄 File contains {len(lines)} line(s). Showing first structure.")
                                return smiles
                            else:
                                st.error("❌ Invalid SMILES in file")
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
