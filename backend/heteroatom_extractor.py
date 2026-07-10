"""
TrackMyPDB Heteroatom Extractor
@author Anu Gamage

This module extracts heteroatoms from PDB structures associated with UniProt proteins.
Licensed under MIT License - Open Source Project
"""

import requests
import pandas as pd
from tqdm import tqdm
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st


class HeteroatomExtractor:
    """
    A comprehensive tool for extracting heteroatoms from PDB structures
    """
    
    def __init__(self):
        # PDBe API endpoint for best structure mappings
        self.PDBe_BEST = "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures"
        self.failed_pdbs = []
        self.all_records = []
        
        # Common non-drug molecules to exclude (water, ions, buffers, solvents)
        self.EXCLUDE_CODES = {
            'HOH', 'WAT', 'H2O', 'DOD', 'D2O',  # Water
            'SO4', 'PO4', 'NO3', 'CL', 'BR', 'I', 'F',  # Common ions
            'NA', 'K', 'CA', 'MG', 'ZN', 'FE', 'CU', 'MN',  # Metal ions
            'ACT', 'EDO', 'PEG', 'GOL', 'MPD', 'DMS', 'BME',  # Common buffers/solvents
            'PGE', 'P6G', 'PE4', 'PE3', 'PE8', '1PE', 'TRS',  # Polyethylene glycols
            'MES', 'EPE', 'IMD', 'ACE', 'IOD', 'CIT'  # More common non-drugs
        }
        
    def get_pdbs_for_uniprot(self, uniprot):
        """
        Get PDB IDs for given UniProt ID from PDBe best mappings
        
        Args:
            uniprot (str): UniProt ID
            
        Returns:
            list: List of PDB IDs
        """
        try:
            r = requests.get(f"{self.PDBe_BEST}/{uniprot}", timeout=10)
            r.raise_for_status()
            data = r.json()
            structs = []
            
            if isinstance(data, dict) and uniprot in data:
                val = data[uniprot]
                if isinstance(val, dict):
                    structs = val.get("best_structures", [])
                elif isinstance(val, list):
                    structs = val
            elif isinstance(data, list):
                structs = data
                
            return sorted({s["pdb_id"].upper() for s in structs if s.get("pdb_id")})
        except Exception as e:
            st.error(f"Error fetching PDBs for {uniprot}: {e}")
            return []

    def download_pdb(self, pdb):
        """
        Download PDB file and return lines
        
        Args:
            pdb (str): PDB ID
            
        Returns:
            list: List of lines from PDB file
        """
        try:
            url = f"https://files.rcsb.org/download/{pdb}.pdb"
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.text.splitlines()
        except Exception as e:
            st.warning(f"Error downloading {pdb}: {e}")
            return []

    def extract_all_heteroatoms(self, lines):
        """
        Extract ALL unique heteroatom codes from HETATM lines
        
        Args:
            lines (list): PDB file lines
            
        Returns:
            tuple: (heteroatom codes list, heteroatom details dict)
        """
        hets = set()
        het_details = {}

        for line in lines:
            if line.startswith("HETATM"):
                # Extract residue name (columns 18-20)
                code = line[17:20].strip()
                if code:  # Only add non-empty codes
                    hets.add(code)

                    # Extract additional info for context
                    if code not in het_details:
                        try:
                            chain = line[21:22].strip()
                            res_num = line[22:26].strip()
                            atom_name = line[12:16].strip()
                            het_details[code] = {
                                'chains': set([chain]) if chain else set(),
                                'residue_numbers': set([res_num]) if res_num else set(),
                                'atom_names': set([atom_name]) if atom_name else set()
                            }
                        except:
                            het_details[code] = {'chains': set(), 'residue_numbers': set(), 'atom_names': set()}
                    else:
                        try:
                            chain = line[21:22].strip()
                            res_num = line[22:26].strip()
                            atom_name = line[12:16].strip()
                            if chain:
                                het_details[code]['chains'].add(chain)
                            if res_num:
                                het_details[code]['residue_numbers'].add(res_num)
                            if atom_name:
                                het_details[code]['atom_names'].add(atom_name)
                        except:
                            pass

        return sorted(list(hets)), het_details

    def fetch_smiles_rcsb(self, code):
        """
        Fetch SMILES from RCSB core chemcomp API
        
        Args:
            code (str): Heteroatom code
            
        Returns:
            dict: Chemical information including SMILES
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{code}"
                r = requests.get(url, timeout=15)

                if r.status_code == 200:
                    data = r.json()
                    smiles = data.get("rcsb_chem_comp_descriptor", {}).get("smiles", "")
                    chem_name = data.get("chem_comp", {}).get("name", "")
                    formula = data.get("chem_comp", {}).get("formula", "")
                    return {
                        'smiles': smiles,
                        'name': chem_name,
                        'formula': formula,
                        'status': 'success' if smiles else 'no_smiles'
                    }
                elif r.status_code == 404:
                    return {'smiles': '', 'name': '', 'formula': '', 'status': 'not_in_rcsb'}
                else:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return {'smiles': '', 'name': '', 'formula': '', 'status': f'http_{r.status_code}'}

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {'smiles': '', 'name': '', 'formula': '', 'status': 'timeout'}
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {'smiles': '', 'name': '', 'formula': '', 'status': f'error_{str(e)[:20]}'}

        return {'smiles': '', 'name': '', 'formula': '', 'status': 'failed_all_retries'}

    def fetch_from_pubchem(self, code):
        """
        Try to fetch SMILES from PubChem as backup
        
        Args:
            code (str): Heteroatom code
            
        Returns:
            str: SMILES string or empty string
        """
        try:
            # Try by compound name
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{code}/property/CanonicalSMILES/JSON"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                props = data.get("PropertyTable", {}).get("Properties", [])
                if props and len(props) > 0:
                    return props[0].get("CanonicalSMILES", "")
        except:
            pass
        return ""
    
    def fetch_smiles_pdb_ligand_expo(self, code):
        """
        Fetch SMILES from PDB Chemical Component Dictionary (Ligand Expo)
        This is the authoritative source for all PDB ligands including 8D0
        
        Args:
            code (str): 3-letter heteroatom code
            
        Returns:
            dict: Chemical information including SMILES
        """
        try:
            # Try ideal coordinates SDF format first (most reliable)
            url = f"https://files.wwpdb.org/pub/pdb/data/monomers/{code[0].lower()}/{code}/{code}_ideal.sdf"
            r = requests.get(url, timeout=15)
            
            if r.status_code == 200:
                # Parse SDF to get SMILES using RDKit
                try:
                    from rdkit import Chem
                    mol = Chem.MolFromMolBlock(r.text)
                    if mol:
                        smiles = Chem.MolToSmiles(mol)
                        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
                        return {
                            'smiles': smiles,
                            'name': code,  # Will be enriched later
                            'formula': formula,
                            'status': 'pdb_ligand_expo_sdf'
                        }
                except ImportError:
                    st.warning("RDKit not available for SDF parsing")
                except Exception as e:
                    st.warning(f"Error parsing SDF for {code}: {e}")
            
            # Alternative: Try model coordinates
            url_model = f"https://files.wwpdb.org/pub/pdb/data/monomers/{code[0].lower()}/{code}/{code}_model.sdf"
            r_model = requests.get(url_model, timeout=15)
            
            if r_model.status_code == 200:
                try:
                    from rdkit import Chem
                    mol = Chem.MolFromMolBlock(r_model.text)
                    if mol:
                        smiles = Chem.MolToSmiles(mol)
                        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
                        return {
                            'smiles': smiles,
                            'name': code,
                            'formula': formula,
                            'status': 'pdb_ligand_expo_model'
                        }
                except Exception as e:
                    pass
        
        except Exception as e:
            pass
        
        return {'smiles': '', 'name': '', 'formula': '', 'status': 'pdb_ligand_expo_failed'}
    
    def fetch_smiles_enhanced(self, code):
        """
        Enhanced SMILES fetching with multiple fallbacks:
        1. RCSB chemcomp API (fastest, most metadata)
        2. PDB Ligand Expo (authoritative for PDB ligands like 8D0)
        3. PubChem (broad coverage)
        
        Args:
            code (str): Heteroatom code
            
        Returns:
            dict: Chemical information including SMILES
        """
        # Try RCSB first (fastest and has metadata)
        rcsb_result = self.fetch_smiles_rcsb(code)
        if rcsb_result['smiles']:
            return rcsb_result
        
        # Try PDB Ligand Expo (authoritative for PDB ligands like 8D0)
        st.info(f"RCSB failed for {code}, trying PDB Ligand Expo...")
        pdb_result = self.fetch_smiles_pdb_ligand_expo(code)
        if pdb_result['smiles']:
            st.success(f"✅ Found {code} in PDB Ligand Expo!")
            return pdb_result
        
        # Try PubChem as last resort
        st.info(f"PDB Ligand Expo failed for {code}, trying PubChem...")
        pubchem_smiles = self.fetch_from_pubchem(code)
        if pubchem_smiles:
            return {
                'smiles': pubchem_smiles,
                'name': code,
                'formula': '',
                'status': 'pubchem_fallback'
            }
        
        # All methods failed
        return {
            'smiles': '',
            'name': '',
            'formula': '',
            'status': 'all_sources_failed'
        }

    def process_pdb_heteroatoms(self, pdb_id, uniprot_id, lines):
        """
        Process all heteroatoms from a single PDB (excluding common non-drug molecules)
        
        Args:
            pdb_id (str): PDB ID
            uniprot_id (str): UniProt ID
            lines (list): PDB file lines
            
        Returns:
            list: List of heteroatom records
        """
        codes, het_details = self.extract_all_heteroatoms(lines)
        
        # Filter out common non-drug molecules
        codes = [c for c in codes if c.upper() not in self.EXCLUDE_CODES]
        
        results = []

        if not codes:
            results.append({
                "UniProt_ID": uniprot_id,
                "PDB_ID": pdb_id,
                "Heteroatom_Code": "NO_DRUG_HETEROATOMS",
                "SMILES": "",
                "Chemical_Name": "",
                "Formula": "",
                "Status": "no_drug_heteroatoms_after_filter",
                "Chains": "",
                "Residue_Numbers": "",
                "Atom_Count": 0
            })
            return results

        st.info(f"Processing {len(codes)} drug-like heteroatoms from {pdb_id}: {', '.join(codes)}")

        for code in codes:
            # Get detailed info
            details = het_details.get(code, {})
            chains = ', '.join(sorted(details.get('chains', set())))
            res_nums = ', '.join(sorted(details.get('residue_numbers', set())))
            atom_count = len(details.get('atom_names', set()))

            # Fetch SMILES using enhanced method with multiple fallbacks
            rcsb_data = self.fetch_smiles_enhanced(code)
            smiles = rcsb_data['smiles']

            results.append({
                "UniProt_ID": uniprot_id,
                "PDB_ID": pdb_id,
                "Heteroatom_Code": code,
                "SMILES": smiles,
                "Chemical_Name": rcsb_data['name'],
                "Formula": rcsb_data['formula'],
                "Status": rcsb_data['status'],
                "Chains": chains,
                "Residue_Numbers": res_nums,
                "Atom_Count": atom_count
            })

            # Small delay to be respectful to APIs
            time.sleep(0.2)

        return results

    def extract_heteroatoms(self, uniprot_ids, progress_callback=None):
        """
        Main function to extract heteroatoms from UniProt IDs
        
        Args:
            uniprot_ids (list): List of UniProt IDs
            progress_callback (function): Optional callback for progress updates
            
        Returns:
            pd.DataFrame: Complete heteroatom data
        """
        self.all_records = []
        self.failed_pdbs = []
        total_heteroatoms = 0

        total_progress = 0
        total_pdbs = 0
        
        # First, count total PDBs for progress tracking
        for up in uniprot_ids:
            pdbs = self.get_pdbs_for_uniprot(up)
            total_pdbs += len(pdbs)

        current_progress = 0

        for up in uniprot_ids:
            pdbs = self.get_pdbs_for_uniprot(up)
            st.info(f"Found {len(pdbs)} PDB structures for {up}")

            for pdb in pdbs:
                try:
                    if progress_callback:
                        progress_callback(current_progress / total_pdbs if total_pdbs > 0 else 0, 
                                       f"Processing {pdb} for {up}")
                    
                    # Download PDB file
                    lines = self.download_pdb(pdb)
                    if not lines:
                        self.failed_pdbs.append(pdb)
                        current_progress += 1
                        continue

                    # Process all heteroatoms
                    pdb_results = self.process_pdb_heteroatoms(pdb, up, lines)
                    self.all_records.extend(pdb_results)

                    # Count heteroatoms found
                    heteroatom_count = len([r for r in pdb_results if r['Heteroatom_Code'] != 'NO_HETEROATOMS'])
                    total_heteroatoms += heteroatom_count

                except Exception as e:
                    st.error(f"Error processing {pdb}: {e}")
                    self.failed_pdbs.append(pdb)
                    
                current_progress += 1

        # Create comprehensive DataFrame
        df = pd.DataFrame(self.all_records)
        
        # Display comprehensive analysis
        st.success("Heteroatom extraction completed!")
        st.write(f"**Total records:** {len(df)}")
        st.write(f"**PDB structures processed:** {df['PDB_ID'].nunique()}")
        st.write(f"**Total unique heteroatoms found:** {df['Heteroatom_Code'].nunique()}")
        st.write(f"**Records with SMILES:** {len(df[df['SMILES'] != ''])}")
        
        if self.failed_pdbs:
            st.warning(f"**Failed PDB downloads:** {len(self.failed_pdbs)}")

        # Show status breakdown
        st.subheader("Status Breakdown")
        status_counts = df['Status'].value_counts()
        st.write(status_counts)

        return df 