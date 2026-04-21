"""
TrackMyPDB - Streamlit Application
@author Anu Gamage

A comprehensive bioinformatics pipeline for extracting heteroatoms from protein structures
and finding molecularly similar compounds using fingerprint-based similarity analysis.

Licensed under MIT License - Open Source Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import sys
import base64
import requests
from datetime import datetime

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import backend modules
try:
    from backend.heteroatom_extractor import HeteroatomExtractor
    # Try to import the full RDKit version first
    try:
        from backend.similarity_analyzer import MolecularSimilarityAnalyzer
        RDKIT_AVAILABLE = True
    except ImportError:
        # Fall back to simplified version
        from backend.similarity_analyzer_simple import MolecularSimilarityAnalyzer
        RDKIT_AVAILABLE = False
        st.warning("⚠️ RDKit not available - using simplified molecular similarity")
except ImportError as e:
    st.error(f"Error importing backend modules: {e}")
    st.stop()

def get_base64_image(image_path):
    """Convert image to base64 string for embedding in HTML"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return ""

def get_pdb_protein_info(pdb_id):
    """
    Fetch protein information (UniProt IDs and protein names) for a given PDB ID
    using RCSB PDB GraphQL API for more reliable data retrieval
    
    Args:
        pdb_id (str): PDB ID
        
    Returns:
        dict: Dictionary with 'uniprot_ids' (list) and 'protein_names' (list)
    """
    try:
        pdb_id = pdb_id.upper()
        
        # Use GraphQL API for more reliable data
        graphql_url = "https://data.rcsb.org/graphql"
        
        query = """
        query ($pdbId: String!) {
          entry(entry_id: $pdbId) {
            polymer_entities {
              rcsb_polymer_entity_container_identifiers {
                reference_sequence_identifiers {
                  database_accession
                  database_name
                }
                uniprot_ids
              }
              entity_poly {
                pdbx_description
              }
              rcsb_polymer_entity {
                pdbx_description
              }
            }
          }
        }
        """
        
        variables = {"pdbId": pdb_id}
        
        response = requests.post(
            graphql_url,
            json={"query": query, "variables": variables},
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        
        if response.status_code != 200:
            return {'uniprot_ids': [], 'protein_names': []}
        
        data = response.json()
        
        uniprot_ids = []
        protein_names = []
        
        # Parse response
        if 'data' in data and data['data'] and 'entry' in data['data']:
            entry = data['data']['entry']
            
            if entry and 'polymer_entities' in entry:
                for entity in entry['polymer_entities']:
                    # Extract UniProt IDs
                    if 'rcsb_polymer_entity_container_identifiers' in entity:
                        identifiers = entity['rcsb_polymer_entity_container_identifiers']
                        
                        # Direct uniprot_ids field
                        if 'uniprot_ids' in identifiers and identifiers['uniprot_ids']:
                            for uid in identifiers['uniprot_ids']:
                                if uid and uid not in uniprot_ids:
                                    uniprot_ids.append(uid)
                        
                        # From reference_sequence_identifiers
                        if 'reference_sequence_identifiers' in identifiers:
                            for ref in identifiers['reference_sequence_identifiers']:
                                if ref.get('database_name') == 'UniProt':
                                    acc = ref.get('database_accession')
                                    if acc and acc not in uniprot_ids:
                                        uniprot_ids.append(acc)
                    
                    # Extract protein descriptions
                    if 'entity_poly' in entity and entity['entity_poly']:
                        desc = entity['entity_poly'].get('pdbx_description')
                        if desc and desc not in protein_names:
                            protein_names.append(desc)
                    
                    if 'rcsb_polymer_entity' in entity and entity['rcsb_polymer_entity']:
                        desc = entity['rcsb_polymer_entity'].get('pdbx_description')
                        if desc and desc not in protein_names:
                            protein_names.append(desc)
        
        return {
            'uniprot_ids': uniprot_ids,
            'protein_names': protein_names
        }
        
    except Exception as e:
        # Log error for debugging
        return {'uniprot_ids': [], 'protein_names': [], 'error': str(e)}

def enrich_results_with_protein_info(results_df):
    """
    Enrich search results with UniProt IDs and protein names
    
    Args:
        results_df (pd.DataFrame): Results dataframe with PDB_ID column
        
    Returns:
        pd.DataFrame: Enriched dataframe with UniProt_IDs and Protein_Names columns
    """
    unique_pdbs = results_df['PDB_ID'].unique()
    
    # Create a cache for PDB information
    pdb_info_cache = {}
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    status_container = st.container()
    
    success_count = 0
    error_count = 0
    
    for idx, pdb_id in enumerate(unique_pdbs):
        progress_text.text(f"Fetching protein information for {pdb_id} ({idx + 1}/{len(unique_pdbs)})...")
        progress_bar.progress((idx + 1) / len(unique_pdbs))
        
        try:
            info = get_pdb_protein_info(pdb_id)
            pdb_info_cache[pdb_id] = info
            
            # Check if data was retrieved
            if info.get('uniprot_ids') or info.get('protein_names'):
                success_count += 1
            elif 'error' in info:
                error_count += 1
                with status_container:
                    st.warning(f"⚠️ Error fetching {pdb_id}: {info.get('error', 'Unknown error')}")
            
            # Small delay to avoid rate limiting
            time.sleep(0.15)
        except Exception as e:
            error_count += 1
            pdb_info_cache[pdb_id] = {'uniprot_ids': [], 'protein_names': []}
            with status_container:
                st.warning(f"⚠️ Failed to fetch {pdb_id}: {str(e)}")
    
    progress_text.empty()
    progress_bar.empty()
    
    # Show summary
    with status_container:
        if success_count > 0:
            st.success(f"✅ Successfully fetched protein information for {success_count}/{len(unique_pdbs)} PDB structures")
        if error_count > 0:
            st.info(f"ℹ️ {error_count} PDB structures had no protein information or encountered errors")
    
    # Add columns to results
    results_df['UniProt_IDs'] = results_df['PDB_ID'].apply(
        lambda x: ', '.join(pdb_info_cache.get(x, {}).get('uniprot_ids', [])) if pdb_info_cache.get(x, {}).get('uniprot_ids', []) else 'N/A'
    )
    
    results_df['Protein_Names'] = results_df['PDB_ID'].apply(
        lambda x: ' | '.join(pdb_info_cache.get(x, {}).get('protein_names', [])) if pdb_info_cache.get(x, {}).get('protein_names', []) else 'N/A'
    )
    
    return results_df

# Page configuration
st.set_page_config(
    page_title="TrackMyPDB",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Apple-like design with pale green theme
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
        background-color: #f8fffe;
    }
    .stApp {
        background-color: #f0f9f5;
    }
    .stButton > button {
        background: linear-gradient(45deg, #4CAF50, #81C784);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
    }
    .metric-card {
        background: rgba(244, 255, 250, 0.8);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(129, 199, 132, 0.3);
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2E7D32;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #4CAF50;
        padding-bottom: 0.5rem;
    }
    .watermark {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: rgba(240, 249, 245, 0.95);
        padding: 12px 20px;
        border-radius: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(76, 175, 80, 0.3);
        font-size: 0.85rem;
        color: #2E7D32;
        font-weight: 500;
        z-index: 1000;
        backdrop-filter: blur(10px);
    }
    .linkedin-link {
        color: #2E7D32;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        transition: all 0.3s ease;
    }
    .linkedin-link:hover {
        color: #1B5E20;
        transform: scale(1.05);
    }
    .linkedin-icon {
        width: 16px;
        height: 16px;
        fill: currentColor;
    }
    .footer {
        margin-top: 4rem;
        padding: 2rem 0;
        border-top: 1px solid rgba(76, 175, 80, 0.3);
        text-align: center;
        color: #4CAF50;
        font-size: 0.9rem;
    }
    .sidebar .sidebar-content {
        background-color: #f4fffb;
    }
    .stSelectbox > div > div {
        background-color: #f4fffb;
    }
    .stTextInput > div > div > input {
        background-color: #f4fffb;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    .stTextArea > div > div > textarea {
        background-color: #f4fffb;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
</style>
""", unsafe_allow_html=True)

def show_sidebar_watermark():
    """Display watermark at bottom of sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="
        margin-top: 2rem;
        padding: 1rem;
        background: rgba(244, 255, 250, 0.8);
        border-radius: 12px;
        border: 1px solid rgba(76, 175, 80, 0.3);
        text-align: center;
        font-size: 0.85rem;
        color: #2E7D32;
        backdrop-filter: blur(10px);
    ">
        <strong>Developed and released by Standard Seed Corporation</strong><br>
        <small style="color: #666;">TrackMyPDB V 1.0</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Add SSC logo if it exists
    if os.path.exists("ssc.png"):
        st.sidebar.markdown("""
        <div style="text-align: center; margin-top: 1rem;">
            <img src="data:image/png;base64,{}" width="80" style="
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(76, 175, 80, 0.2);
                margin-top: 0.5rem;
            ">
        </div>
        """.format(get_base64_image("ssc.png")), unsafe_allow_html=True)
    else:
        st.sidebar.info("💡 Add ssc.png to display SSC logo")

def show_footer():
    """Display footer with license information"""
    st.markdown("""
    <div class="footer">
        <p>📄 Licensed under MIT License - Open Source Project</p>
        <p>🧬 TrackMyPDB - Bioinformatics Pipeline for Protein Structure Analysis</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.title("🧬 TrackMyPDB")
    st.markdown("### *Protein Structure Heteroatom Extraction & Molecular Similarity Analysis*")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["🏠 Home", "🔍 Heteroatom Extraction", "🧪 Similarity Analysis", "� SMILES Database Search", "�📊 Complete Pipeline"]
    )
    
    # Add watermark at bottom of sidebar
    show_sidebar_watermark()
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔍 Heteroatom Extraction":
        show_extraction_page()
    elif page == "🧪 Similarity Analysis":
        show_similarity_page()
    elif page == "� SMILES Database Search":
        show_smiles_database_search()
    elif page == "�📊 Complete Pipeline":
        show_complete_pipeline()
    
    # Show footer
    show_footer()

def show_home_page():
    """Display home page with project overview"""
    
    st.markdown('<div class="section-header">🎯 Project Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🔬 Heteroatom Extraction Tool
        - **Purpose**: Extract ALL heteroatoms from PDB structures
        - **Input**: UniProt protein identifiers
        - **Output**: Comprehensive ligand database with SMILES
        - **Features**: Multi-source data fetching, robust error handling
        """)
        
        st.markdown("""
        #### 📊 Key Capabilities
        - ✅ **Comprehensive extraction**: Processes ALL heteroatoms
        - ✅ **Multi-source data**: RCSB PDB and PubChem APIs
        - ✅ **Progress tracking**: Real-time status updates
        - ✅ **Error handling**: Graceful API failure management
        """)
    
    with col2:
        st.markdown("""
        #### 🧪 Molecular Similarity Analyzer
        - **Purpose**: Find molecules similar to target compound
        - **Input**: Target SMILES structure
        - **Output**: Ranked similarity results
        - **Method**: Morgan fingerprints + Tanimoto similarity
        """)
        
        st.markdown("""
        #### 🎯 Analysis Features
        - ✅ **Morgan fingerprints**: Industry-standard representations
        - ✅ **Tanimoto similarity**: Robust similarity metrics
        - ✅ **Rich visualizations**: Interactive plots and charts
        - ✅ **Statistical reports**: Comprehensive analysis
        """)
    
    # New feature highlight
    st.markdown('<div class="section-header">🆕 SMILES Database Search</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🔬 Novel Search Feature
        - **Purpose**: Search pre-built PDB ligands database
        - **Input**: SMILES structure only (no UniProt needed)
        - **Output**: Top matching PDB IDs with similarity scores
        - **Database**: pdb_ligands_trackmypdb_open_source.csv
        """)
    
    with col2:
        st.markdown("""
        #### ⚡ Search Benefits
        - ✅ **Fast search**: No extraction needed
        - ✅ **Direct matching**: Query against all PDB ligands
        - ✅ **Top results**: Best co-crystallized ligands
        - ✅ **PDB annotations**: Direct PDB ID mapping
        """)
    
    # Workflow diagram
    st.markdown('<div class="section-header">🔄 Workflows</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Standard Pipeline:**
        ```
        UniProt IDs 
           ↓
        PDB Structures 
           ↓
        Heteroatom Extraction 
           ↓
        SMILES Database
           ↓
        Target SMILES 
           ↓
        Similarity Analysis 
           ↓
        Results CSV
        ```
        """)
    
    with col2:
        st.markdown("""
        **SMILES Database Search:**
        ```
        Input SMILES
           ↓
        Morgan Fingerprints
           ↓
        PDB Ligands Database
           ↓
        Tanimoto Similarity
           ↓
        Top PDB IDs
           ↓
        Annotated Results
        ```
        """)
    
    # Quick start
    st.markdown('<div class="section-header">🚀 Quick Start</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Option 1: Full Pipeline**
        1. Navigate to "🔍 Heteroatom Extraction"
        2. Enter your UniProt IDs
        3. Run extraction to build database
        4. Switch to "🧪 Similarity Analysis"
        5. Input your target SMILES
        6. Analyze molecular similarities
        7. Download results as CSV
        """)
    
    with col2:
        st.markdown("""
        **Option 2: Quick SMILES Search** ⚡
        1. Navigate to "🔬 SMILES Database Search"
        2. Enter your SMILES structure(s)
        3. Set search parameters
        4. Click "Search Database"
        5. View top matching PDB IDs
        6. Download results
        7. Analyze co-crystallized ligands
        """)

def show_extraction_page():
    """Display heteroatom extraction interface"""
    
    st.markdown('<div class="section-header">🔍 Heteroatom Extraction</div>', unsafe_allow_html=True)
    
    # Input section
    st.subheader("📋 Input Parameters")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # UniProt IDs input
        uniprot_input = st.text_area(
            "UniProt IDs",
            placeholder="Enter UniProt IDs (one per line or comma-separated)\nExample: Q9UNQ0, P37231, P06276",
            height=100,
            help="Enter protein UniProt identifiers to extract heteroatoms from associated PDB structures"
        )
        
        # Parse UniProt IDs
        if uniprot_input:
            # Handle both comma-separated and line-separated input
            uniprot_ids = []
            for line in uniprot_input.strip().split('\n'):
                for up_id in line.split(','):
                    up_id = up_id.strip()
                    if up_id:
                        uniprot_ids.append(up_id)
            
            st.info(f"Found {len(uniprot_ids)} UniProt IDs: {', '.join(uniprot_ids)}")
    
    with col2:
        st.markdown("#### 📊 Extraction Settings")
        
        # Download existing results
        if os.path.exists("heteroatom_results.csv"):
            st.success("Previous results found!")
            if st.button("📥 Load Previous Results"):
                df = pd.read_csv("heteroatom_results.csv")
                st.session_state['heteroatom_data'] = df
                st.success("Previous results loaded!")
        
        # Clear results
        if st.button("🗑️ Clear Results"):
            if 'heteroatom_data' in st.session_state:
                del st.session_state['heteroatom_data']
            if os.path.exists("heteroatom_results.csv"):
                os.remove("heteroatom_results.csv")
            st.success("Results cleared!")
    
    # Run extraction
    if st.button("🚀 Start Heteroatom Extraction", type="primary"):
        if not uniprot_input:
            st.error("Please enter at least one UniProt ID")
            return
        
        # Initialize extractor
        extractor = HeteroatomExtractor()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress, message):
            progress_bar.progress(progress)
            status_text.text(message)
        
        try:
            # Run extraction
            with st.spinner("Extracting heteroatoms..."):
                df = extractor.extract_heteroatoms(uniprot_ids, progress_callback=update_progress)
            
            # Store results
            st.session_state['heteroatom_data'] = df
            
            # Save to CSV
            df.to_csv("heteroatom_results.csv", index=False)
            
            progress_bar.empty()
            status_text.empty()
            
            st.success("Heteroatom extraction completed successfully!")
            
        except Exception as e:
            st.error(f"Error during extraction: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    
    # Display results
    if 'heteroatom_data' in st.session_state:
        df = st.session_state['heteroatom_data']
        
        st.markdown('<div class="section-header">📊 Extraction Results</div>', unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("PDB Structures", df['PDB_ID'].nunique())
        with col3:
            st.metric("Unique Heteroatoms", df['Heteroatom_Code'].nunique())
        with col4:
            st.metric("Records with SMILES", len(df[df['SMILES'] != '']))
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Results (CSV)",
            data=csv,
            file_name=f"heteroatom_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def show_similarity_page():
    """Display molecular similarity analysis interface"""
    
    st.markdown('<div class="section-header">🧪 Molecular Similarity Analysis</div>', unsafe_allow_html=True)
    
    # Check if heteroatom data exists
    if 'heteroatom_data' not in st.session_state:
        if os.path.exists("heteroatom_results.csv"):
            df = pd.read_csv("heteroatom_results.csv")
            st.session_state['heteroatom_data'] = df
            st.success("Loaded previous heteroatom extraction results!")
        else:
            st.warning("No heteroatom data found. Please run heteroatom extraction first.")
            return
    
    # Input section
    st.subheader("🎯 Target Molecule")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        target_smiles = st.text_input(
            "Target SMILES Structure",
            placeholder="Enter SMILES string (e.g., CCO for ethanol)",
            help="Enter the SMILES representation of your target molecule"
        )
        
        # SMILES validation
        if target_smiles:
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(target_smiles)
                if mol is not None:
                    st.success("✅ Valid SMILES structure")
                else:
                    st.error("❌ Invalid SMILES structure")
            except:
                st.error("❌ Error validating SMILES")
    
    with col2:
        st.markdown("#### ⚙️ Analysis Parameters")
        
        top_n = st.slider("Number of Results", 10, 100, 50)
        min_similarity = st.slider("Minimum Similarity", 0.0, 1.0, 0.2, 0.1)
        
        st.markdown("#### 📊 Fingerprint Settings")
        radius = st.selectbox("Morgan Radius", [1, 2, 3], index=1)
        n_bits = st.selectbox("Fingerprint Bits", [1024, 2048, 4096], index=1)
    
    # Run analysis
    if st.button("🔍 Analyze Molecular Similarity", type="primary"):
        if not target_smiles:
            st.error("Please enter a target SMILES structure")
            return
        
        # Initialize analyzer
        analyzer = MolecularSimilarityAnalyzer(radius=radius, n_bits=n_bits)
        
        try:
            # Run analysis
            with st.spinner("Analyzing molecular similarity..."):
                heteroatom_df = st.session_state['heteroatom_data']
                similarity_results = analyzer.analyze_similarity(
                    target_smiles=target_smiles,
                    heteroatom_df=heteroatom_df,
                    top_n=top_n,
                    min_similarity=min_similarity
                )
            
            # Store results
            st.session_state['similarity_results'] = similarity_results
            
            # Enhanced download functionality
            if not similarity_results.empty:
                # Prepare the CSV with exactly the columns requested: 
                # PDB_ID, Heteroatom_Code, Chemical_Name, SMILES, Tanimoto_Similarity, Formula
                download_df = similarity_results[[
                    'PDB_ID', 
                    'Heteroatom_Code', 
                    'Chemical_Name', 
                    'SMILES', 
                    'Tanimoto_Similarity', 
                    'Formula'
                ]].copy()
                
                # Format similarity scores
                download_df['Tanimoto_Similarity'] = download_df['Tanimoto_Similarity'].round(4)
                
                # Sort by best Tanimoto scores (highest first)
                download_df = download_df.sort_values('Tanimoto_Similarity', ascending=False).reset_index(drop=True)
                
                # Save to CSV
                download_df.to_csv("similarity_results.csv", index=False)
                
                # Enhanced download section
                st.markdown("---")
                st.subheader("📥 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("📊 Total Results", len(download_df))
                    st.metric("🏆 Best Score", f"{download_df['Tanimoto_Similarity'].max():.4f}")
                
                with col2:
                    # Download button with enhanced CSV
                    csv_data = download_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Complete Similarity Results (CSV)",
                        data=csv_data,
                        file_name=f"TrackMyPDB_similarity_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Downloads the complete similarity analysis results sorted by best Tanimoto scores"
                    )
                
                # Show preview of downloadable data
                st.subheader("📋 Download Preview (Top 10 Results)")
                st.dataframe(
                    download_df.head(10),
                    use_container_width=True,
                    column_config={
                        "PDB_ID": "PDB ID",
                        "Heteroatom_Code": "Ligand Code",
                        "Chemical_Name": "Chemical Name", 
                        "SMILES": st.column_config.TextColumn("SMILES", width="medium"),
                        "Tanimoto_Similarity": st.column_config.NumberColumn(
                            "Similarity Score",
                            help="Higher scores indicate better matches",
                            format="%.4f"
                        ),
                        "Formula": "Molecular Formula"
                    },
                    hide_index=True
                )
            
            else:
                st.warning("⚠️ No results found above the similarity threshold. Try lowering the minimum similarity value.")
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

def show_smiles_database_search():
    """Display SMILES database search interface for finding similar PDB ligands"""
    
    st.markdown('<div class="section-header">🔬 SMILES Database Search</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Search the PDB ligands database using your SMILES structure. This tool will:
    1. Generate Morgan fingerprints for your input SMILES
    2. Compare against all co-crystallized ligands in the database
    3. Return top matching PDB IDs with their similarity scores
    """)
    
    # Check if database exists
    db_path = "pdb_ligands_trackmypdb_open_source.csv"
    if not os.path.exists(db_path):
        st.error(f"❌ Database file '{db_path}' not found. Please ensure the file exists in the application directory.")
        return
    
    # Input section
    st.subheader("🎯 Input SMILES")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        target_smiles = st.text_area(
            "Target SMILES Structure",
            placeholder="Enter SMILES string (e.g., CCO for ethanol)\nYou can enter multiple SMILES, one per line",
            height=120,
            help="Enter the SMILES representation of your target molecule(s)"
        )
        
        # SMILES validation
        if target_smiles:
            smiles_list = [s.strip() for s in target_smiles.strip().split('\n') if s.strip()]
            
            if RDKIT_AVAILABLE:
                try:
                    from rdkit import Chem
                    valid_count = 0
                    for smiles in smiles_list:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            valid_count += 1
                    
                    if valid_count == len(smiles_list):
                        st.success(f"✅ All {len(smiles_list)} SMILES structure(s) are valid")
                    else:
                        st.warning(f"⚠️ {valid_count}/{len(smiles_list)} SMILES structure(s) are valid")
                except Exception as e:
                    st.error(f"❌ Error validating SMILES: {str(e)}")
            else:
                st.info(f"Found {len(smiles_list)} SMILES structure(s)")
    
    with col2:
        st.markdown("#### ⚙️ Search Parameters")
        
        top_n = st.slider("Number of Top Results", 5, 100, 20, help="Number of top matching PDB IDs to return")
        min_similarity = st.slider("Minimum Similarity Threshold", 0.0, 1.0, 0.3, 0.05, help="Minimum Tanimoto similarity score (0-1)")
        
        st.markdown("#### 📊 Fingerprint Settings")
        radius = st.selectbox("Morgan Fingerprint Radius", [1, 2, 3], index=1, help="Radius for Morgan fingerprint generation")
        n_bits = st.selectbox("Fingerprint Bits", [1024, 2048, 4096], index=1, help="Number of bits in fingerprint")
    
    # Database info
    st.subheader("📚 Database Information")
    
    try:
        db_df = pd.read_csv(db_path)
        
        # Filter out entries without SMILES
        db_df_valid = db_df[db_df['SMILES'].notna() & (db_df['SMILES'] != '')]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Ligands", len(db_df))
        with col2:
            st.metric("Valid SMILES", len(db_df_valid))
        with col3:
            st.metric("Unique PDB IDs", db_df['PDB_ID'].nunique())
        with col4:
            st.metric("Unique Ligands", db_df['Heteroatom_Code'].nunique())
        
    except Exception as e:
        st.error(f"Error loading database: {str(e)}")
        return
    
    # Run search
    if st.button("🔍 Search Database", type="primary"):
        if not target_smiles:
            st.error("Please enter at least one target SMILES structure")
            return
        
        # Parse SMILES input
        smiles_list = [s.strip() for s in target_smiles.strip().split('\n') if s.strip()]
        
        try:
            # Load database
            with st.spinner("Loading database..."):
                db_df = pd.read_csv(db_path)
                
                # Filter out entries without SMILES
                db_df = db_df[db_df['SMILES'].notna() & (db_df['SMILES'] != '')]
                
                if len(db_df) == 0:
                    st.error("No valid SMILES found in database")
                    return
                
                st.info(f"Loaded {len(db_df)} ligands with valid SMILES from database")
            
            # Initialize analyzer
            analyzer = MolecularSimilarityAnalyzer(radius=radius, n_bits=n_bits)
            
            # Process each input SMILES
            all_results = []
            
            for idx, query_smiles in enumerate(smiles_list):
                with st.spinner(f"Analyzing SMILES {idx + 1}/{len(smiles_list)}: {query_smiles[:50]}..."):
                    
                    # Generate fingerprint for query SMILES
                    query_fp = analyzer.smiles_to_fingerprint(query_smiles)
                    
                    if query_fp is None:
                        st.warning(f"⚠️ Invalid SMILES (skipped): {query_smiles}")
                        continue
                    
                    # Calculate similarities
                    similarities = []
                    
                    progress_bar = st.progress(0)
                    
                    for i, row in db_df.iterrows():
                        # Generate fingerprint for database SMILES
                        db_fp = analyzer.smiles_to_fingerprint(row['SMILES'])
                        
                        if db_fp is not None:
                            # Calculate Tanimoto similarity
                            similarity = analyzer.calculate_tanimoto_similarity(query_fp, db_fp)
                            
                            if similarity >= min_similarity:
                                similarities.append({
                                    'Query_SMILES': query_smiles,
                                    'PDB_ID': row['PDB_ID'],
                                    'Heteroatom_Code': row['Heteroatom_Code'],
                                    'Chemical_Name': row['Chemical_Name'],
                                    'Database_SMILES': row['SMILES'],
                                    'Tanimoto_Similarity': similarity,
                                    'Formula': row['Formula'],
                                    'Status': row.get('Status', '')
                                })
                        
                        # Update progress
                        if i % 100 == 0:
                            progress_bar.progress(min((i + 1) / len(db_df), 1.0))
                    
                    progress_bar.empty()
                    
                    # Sort by similarity and get top N
                    if similarities:
                        df_results = pd.DataFrame(similarities)
                        df_results = df_results.sort_values('Tanimoto_Similarity', ascending=False)
                        df_results = df_results.head(top_n)
                        all_results.append(df_results)
                        
                        st.success(f"✅ Found {len(similarities)} matches for SMILES {idx + 1} (showing top {min(top_n, len(similarities))})")
                    else:
                        st.warning(f"⚠️ No matches found above similarity threshold {min_similarity} for SMILES {idx + 1}")
            
            # Combine all results
            if all_results:
                final_results = pd.concat(all_results, ignore_index=True)
                
                # Sort by similarity score
                final_results = final_results.sort_values('Tanimoto_Similarity', ascending=False).reset_index(drop=True)
                
                # Store results
                st.session_state['smiles_search_results'] = final_results
                
                st.success(f"🎉 Search completed! Found {len(final_results)} total matches")
                
                # Display results
                st.markdown('<div class="section-header">📊 Search Results</div>', unsafe_allow_html=True)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Matches", len(final_results))
                with col2:
                    st.metric("Best Similarity", f"{final_results['Tanimoto_Similarity'].max():.4f}")
                with col3:
                    st.metric("Unique PDB IDs", final_results['PDB_ID'].nunique())
                with col4:
                    st.metric("Avg Similarity", f"{final_results['Tanimoto_Similarity'].mean():.4f}")
                
                # Results table
                st.subheader("📋 Top Matches")
                
                # Display with better formatting
                display_df = final_results.copy()
                display_df['Tanimoto_Similarity'] = display_df['Tanimoto_Similarity'].round(4)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    column_config={
                        "Query_SMILES": st.column_config.TextColumn("Query SMILES", width="medium"),
                        "PDB_ID": "PDB ID",
                        "Heteroatom_Code": "Ligand Code",
                        "Chemical_Name": "Chemical Name",
                        "Database_SMILES": st.column_config.TextColumn("Database SMILES", width="medium"),
                        "Tanimoto_Similarity": st.column_config.NumberColumn(
                            "Similarity Score",
                            help="Tanimoto similarity score (0-1, higher is better)",
                            format="%.4f"
                        ),
                        "Formula": "Molecular Formula",
                        "Status": "Status"
                    },
                    hide_index=True
                )
            
            else:
                st.error("No matches found for any of the input SMILES structures")
                
        except Exception as e:
            st.error(f"Error during search: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    # Protein Information Enrichment Section (outside search execution)
    # This section persists across reruns using session state
    if 'smiles_search_results' in st.session_state and len(st.session_state['smiles_search_results']) > 0:
        final_results = st.session_state['smiles_search_results']
        
        st.markdown("---")
        st.markdown('<div class="section-header">🧬 Protein Target Information</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Fetch additional protein information for the PDB structures found in your search results.
        This will query the RCSB PDB database to retrieve:
        - **UniProt IDs** associated with each PDB structure
        - **Protein names/descriptions** for each target
        """)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            fetch_button = st.button("🔍 Fetch Protein Information", type="primary")
        
        with col2:
            if 'enriched_results' in st.session_state:
                if st.button("🗑️ Clear Protein Info"):
                    if 'enriched_results' in st.session_state:
                        del st.session_state['enriched_results']
                    st.rerun()
        
        # Fetch protein information if button was clicked
        if fetch_button:
            with st.spinner("Fetching protein information from RCSB PDB..."):
                enriched_df = enrich_results_with_protein_info(final_results.copy())
                st.session_state['enriched_results'] = enriched_df
        
        # Display enriched results if available
        if 'enriched_results' in st.session_state:
            enriched_df = st.session_state['enriched_results']
            
            st.subheader("📋 Enriched Results with Protein Information")
            
            # Display enriched table
            display_enriched = enriched_df.copy()
            display_enriched['Tanimoto_Similarity'] = display_enriched['Tanimoto_Similarity'].round(4)
            
            st.dataframe(
                display_enriched,
                use_container_width=True,
                column_config={
                    "Query_SMILES": st.column_config.TextColumn("Query SMILES", width="small"),
                    "PDB_ID": "PDB ID",
                    "Heteroatom_Code": "Ligand",
                    "Chemical_Name": "Ligand Name",
                    "Database_SMILES": st.column_config.TextColumn("SMILES", width="small"),
                    "Tanimoto_Similarity": st.column_config.NumberColumn(
                        "Similarity",
                        format="%.4f"
                    ),
                    "Formula": "Formula",
                    "Status": "Status",
                    "UniProt_IDs": st.column_config.TextColumn("UniProt IDs", width="medium"),
                    "Protein_Names": st.column_config.TextColumn("Protein Names", width="large")
                },
                hide_index=True
            )
            
            # Summary of protein targets
            st.markdown("---")
            st.subheader("📊 Protein Target Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Count unique proteins
                unique_proteins = enriched_df[enriched_df['UniProt_IDs'] != 'N/A']['UniProt_IDs'].unique()
                st.metric("Unique Protein Targets", len(unique_proteins))
                
                # Show list of UniProt IDs
                if len(unique_proteins) > 0:
                    st.markdown("**UniProt IDs Found:**")
                    all_uniprots = set()
                    for ids in unique_proteins:
                        if ids != 'N/A':
                            all_uniprots.update([uid.strip() for uid in ids.split(',')])
                    
                    for uid in sorted(all_uniprots):
                        st.markdown(f"- [{uid}](https://www.uniprot.org/uniprotkb/{uid})")
            
            with col2:
                # Show protein names
                unique_names = enriched_df[enriched_df['Protein_Names'] != 'N/A']['Protein_Names'].unique()
                st.metric("Unique Protein Names", len(unique_names))
                
                if len(unique_names) > 0:
                    st.markdown("**Protein Names:**")
                    all_names = set()
                    for names in unique_names:
                        if names != 'N/A':
                            all_names.update([n.strip() for n in names.split('|')])
                    
                    for name in sorted(all_names)[:10]:  # Show top 10
                        st.markdown(f"- {name}")
                    
                    if len(all_names) > 10:
                        st.markdown(f"*...and {len(all_names) - 10} more*")
        
        # Download section
        st.markdown("---")
        st.subheader("📥 Download Results")
        
        # Determine which dataframe to use for downloads
        download_df = st.session_state.get('enriched_results', final_results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Full results
            csv_data = download_df.to_csv(index=False)
            label = "📥 Download Enriched Results (CSV)" if 'enriched_results' in st.session_state else "📥 Download Complete Results (CSV)"
            st.download_button(
                label=label,
                data=csv_data,
                file_name=f"TrackMyPDB_SMILES_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download all search results with complete information"
            )
        
        with col2:
            # PDB IDs only
            if 'enriched_results' in st.session_state:
                pdb_protein_df = download_df[['PDB_ID', 'UniProt_IDs', 'Protein_Names', 'Tanimoto_Similarity']].drop_duplicates()
                pdb_protein_csv = pdb_protein_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download PDB-Protein Mapping (CSV)",
                    data=pdb_protein_csv,
                    file_name=f"TrackMyPDB_PDB_Protein_Mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download PDB IDs with associated UniProt IDs and protein names"
                )
            else:
                pdb_ids_df = download_df[['PDB_ID', 'Tanimoto_Similarity']].drop_duplicates()
                pdb_ids_csv = pdb_ids_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download PDB IDs Only (CSV)",
                    data=pdb_ids_csv,
                    file_name=f"TrackMyPDB_PDB_IDs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download unique PDB IDs with their best similarity scores"
                )
        
        # Visualizations (if RDKit available)
        if RDKIT_AVAILABLE and len(final_results) > 0:
            st.markdown('<div class="section-header">📈 Similarity Distribution</div>', unsafe_allow_html=True)
            
            try:
                import plotly.express as px
                import plotly.graph_objects as go
                
                # Histogram of similarity scores
                fig = px.histogram(
                    final_results,
                    x='Tanimoto_Similarity',
                    nbins=30,
                    title='Distribution of Tanimoto Similarity Scores',
                    labels={'Tanimoto_Similarity': 'Tanimoto Similarity Score'},
                    color_discrete_sequence=['#4CAF50']
                )
                fig.update_layout(
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Top 20 matches bar chart
                if len(final_results) >= 10:
                    top_20 = final_results.head(20).copy()
                    top_20['Label'] = top_20['PDB_ID'] + ' - ' + top_20['Heteroatom_Code']
                    
                    fig2 = px.bar(
                        top_20,
                        x='Tanimoto_Similarity',
                        y='Label',
                        orientation='h',
                        title='Top 20 Matches by Similarity Score',
                        labels={'Tanimoto_Similarity': 'Tanimoto Similarity Score', 'Label': 'PDB ID - Ligand'},
                        color='Tanimoto_Similarity',
                        color_continuous_scale='Greens'
                    )
                    fig2.update_layout(
                        showlegend=False,
                        yaxis={'categoryorder': 'total ascending'},
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not generate visualizations: {str(e)}")

def show_complete_pipeline():
    """Display complete pipeline interface"""
    
    st.markdown('<div class="section-header">📊 Complete Pipeline</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Run the complete TrackMyPDB pipeline in one go:
    1. Extract heteroatoms from UniProt proteins
    2. Analyze molecular similarity to target compound
    3. Generate comprehensive results
    """)
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 UniProt Input")
        uniprot_input = st.text_area(
            "UniProt IDs",
            placeholder="Q9UNQ0, P37231, P06276",
            height=100
        )
    
    with col2:
        st.subheader("🎯 Target Molecule")
        target_smiles = st.text_input(
            "Target SMILES",
            placeholder="CCO"
        )
    
    # Parameters
    st.subheader("⚙️ Analysis Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_n = st.slider("Top Results", 10, 100, 50)
    with col2:
        min_similarity = st.slider("Min Similarity", 0.0, 1.0, 0.2, 0.1)
    with col3:
        radius = st.selectbox("Morgan Radius", [1, 2, 3], index=1)
    
    # Run complete pipeline
    if st.button("🚀 Run Complete Pipeline", type="primary"):
        if not uniprot_input or not target_smiles:
            st.error("Please provide both UniProt IDs and target SMILES")
            return
        
        # Parse UniProt IDs
        uniprot_ids = []
        for line in uniprot_input.strip().split('\n'):
            for up_id in line.split(','):
                up_id = up_id.strip()
                if up_id:
                    uniprot_ids.append(up_id)
        
        # Step 1: Heteroatom Extraction
        st.info("Step 1: Extracting heteroatoms...")
        extractor = HeteroatomExtractor()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress, message):
            progress_bar.progress(progress * 0.7)  # Use 70% for extraction
            status_text.text(f"Extraction: {message}")
        
        try:
            heteroatom_df = extractor.extract_heteroatoms(uniprot_ids, progress_callback=update_progress)
            
            # Step 2: Similarity Analysis
            status_text.text("Step 2: Analyzing molecular similarity...")
            progress_bar.progress(0.7)
            
            analyzer = MolecularSimilarityAnalyzer(radius=radius)
            similarity_results = analyzer.analyze_similarity(
                target_smiles=target_smiles,
                heteroatom_df=heteroatom_df,
                top_n=top_n,
                min_similarity=min_similarity
            )
            
            progress_bar.progress(1.0)
            status_text.text("Pipeline completed successfully!")
            
            # Save results
            heteroatom_df.to_csv("complete_pipeline_heteroatoms.csv", index=False)
            if not similarity_results.empty:
                similarity_results.to_csv("complete_pipeline_similarity.csv", index=False)
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                csv1 = heteroatom_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Heteroatom Results",
                    data=csv1,
                    file_name=f"heteroatoms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                if not similarity_results.empty:
                    csv2 = similarity_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Similarity Results",
                        data=csv2,
                        file_name=f"similarity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"Pipeline error: {str(e)}")
            progress_bar.empty()
            status_text.empty()

if __name__ == "__main__":
    main() 