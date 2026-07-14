"""
TrackMyPDB - Streamlit Application
@author Anu Gamage

A comprehensive bioinformatics pipeline for extracting heteroatoms from protein structures
and finding molecularly similar compounds using fingerprint-based similarity analysis.

Licensed under MIT License - Open Source Project
Version: 2.0 - Updated with author names and citation section
"""

# ============================================================================
# CRITICAL FIX: Force matplotlib to use non-GUI backend BEFORE any imports
# This fixes libXrender.so.1 error on Linux/Streamlit Cloud/Docker
# MUST be at the very top before importing any modules that use matplotlib
# ============================================================================
import matplotlib
matplotlib.use('Agg')

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import sys
import base64
import requests
from datetime import datetime
from io import BytesIO

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import backend modules
try:
    from backend.heteroatom_extractor import HeteroatomExtractor
    from backend.molecule_visualizer import MoleculeVisualizer, ChemicalDrawingTool
    from backend.disease_annotator import DiseaseAnnotator, DISEASE_CATEGORIES
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
        
        # Check for errors in response
        if 'errors' in data:
            return {'uniprot_ids': [], 'protein_names': [], 'error': str(data['errors'])}
        
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
        <small style="color: #666;">TrackMyPDB V 2.0</small><br>
        <small style="color: #666; margin-top: 0.5rem; display: block;">
            Suliman Sharif, Anu Gamage, Kalana Kotawalagedara,<br>
            Sakeer Sha, Damilola Bodun
        </small>
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
    """Main application function with enhanced navigation"""
    
    # Header
    st.title("🧬 TrackMyPDB")
    st.markdown("### *Protein Structure Heteroatom Extraction & Molecular Similarity Analysis*")
    
    # Initialize session state for navigation if not exists
    if 'nav_page' not in st.session_state:
        st.session_state['nav_page'] = "🏠 Home"
    
    # Sidebar navigation - synced with session state
    st.sidebar.title("Navigation")
    
    # Use session state to control selectbox
    page_options = [
        "🏠 Home", 
        "🔍 Heteroatom Extraction", 
        "🧪 Similarity Analysis", 
        "🔬 SMILES Database Search", 
        "� Legacy Search",
        "🖼️ Molecule Visualizer",
        "🏥 Disease Enrichment",
        "🤖 AI Assistant"
    ]
    
    # Find index of current page
    current_index = page_options.index(st.session_state['nav_page']) if st.session_state['nav_page'] in page_options else 0
    
    # Sidebar selectbox with synchronized state
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        page_options,
        index=current_index,
        key="sidebar_nav"
    )
    
    # Update session state when sidebar changes
    if page != st.session_state['nav_page']:
        st.session_state['nav_page'] = page
        st.rerun()
    
    # Add watermark at bottom of sidebar
    show_sidebar_watermark()
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔍 Heteroatom Extraction":
        show_extraction_page()
    elif page == "🧪 Similarity Analysis":
        show_similarity_page()
    elif page == "🔬 SMILES Database Search":
        show_smiles_database_search()
    elif page == "� Legacy Search":
        show_complete_pipeline()
    elif page == "🖼️ Molecule Visualizer":
        show_molecule_visualizer_page()
    elif page == "🏥 Disease Enrichment":
        show_disease_enrichment_page()
    elif page == "🤖 AI Assistant":
        show_ai_assistant_page()

    # Show footer
    show_footer()


def show_ai_assistant_page():
    """AI Assistant - Claude-powered chat backed by the TrackMyPDB MCP server."""
    from agent.streamlit_chat import render
    render()

def show_home_page():
    """
    Professional Home Page - Enterprise Bioinformatics Dashboard
    Refactored for clean typography, functional app launcher, and organized metadata
    """
    
    # ============================================================================
    # HERO SECTION - Clean, professional header
    # ============================================================================
    st.markdown("""
        <h1 style='color: #2E7D32; font-weight: 700; margin-bottom: 0.5rem; background: transparent;'>
            TrackMyPDB
        </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <h3 style='color: #4A4A4A; font-weight: 400; margin-top: 0; margin-bottom: 0.5rem; background: transparent;'>
            Protein Structure Heteroatom Extraction & Molecular Similarity Analysis
        </h3>
    """, unsafe_allow_html=True)
    
    # Clean version indicator
    st.caption("Version 2.0.1 | Updated: May 15, 2026")
    
    st.markdown("---")
    
    # ============================================================================
    # APPLICATION LAUNCHER GRID - 2x2 functional dashboard
    # ============================================================================
    st.markdown("""
        <h2 style='color: #2E7D32; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem; background: transparent;'>
            Applications
        </h2>
    """, unsafe_allow_html=True)
    
    # Row 1: Visualizer and Legacy Search
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 8px; 
                        border: 1px solid #E0E0E0; height: 200px; margin-bottom: 1rem;'>
                <h4 style='color: #2E7D32; margin-top: 0; background: transparent;'>
                    Molecular Visualization & Properties
                </h4>
                <p style='color: #666; font-size: 0.95rem; line-height: 1.5;'>
                    Input SMILES, search by name, or upload files to interactively 
                    visualize structures and calculate chemical properties.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Launch Visualizer", key="btn_visualizer", use_container_width=True):
            st.session_state['nav_page'] = "🖼️ Molecule Visualizer"
            st.rerun()
    
    with col2:
        st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 8px; 
                        border: 1px solid #E0E0E0; height: 200px; margin-bottom: 1rem;'>
                <h4 style='color: #2E7D32; margin-top: 0; background: transparent;'>
                    Legacy Search
                </h4>
                <p style='color: #666; font-size: 0.95rem; line-height: 1.5;'>
                    Run the end-to-end processing pipeline—from UniProt ID extraction 
                    directly down to molecular similarity filtering—in one automated step.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Launch Legacy Search", key="btn_legacy", use_container_width=True):
            st.session_state['nav_page'] = "� Legacy Search"
            st.rerun()
    
    # Row 2: Database Search and Disease Analysis
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 8px; 
                        border: 1px solid #E0E0E0; height: 200px; margin-bottom: 1rem;'>
                <h4 style='color: #2E7D32; margin-top: 0; background: transparent;'>
                    SMILES Database Search
                </h4>
                <p style='color: #666; font-size: 0.95rem; line-height: 1.5;'>
                    Query structures directly against our pre-built PDB ligand database 
                    using Morgan Fingerprints and Tanimoto metrics without needing a UniProt input.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Launch Database Search", key="btn_db_search", use_container_width=True):
            st.session_state['nav_page'] = "🔬 SMILES Database Search"
            st.rerun()
    
    with col4:
        st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 8px; 
                        border: 1px solid #E0E0E0; height: 200px; margin-bottom: 1rem;'>
                <h4 style='color: #2E7D32; margin-top: 0; background: transparent;'>
                    Disease Enrichment Analysis
                </h4>
                <p style='color: #666; font-size: 0.95rem; line-height: 1.5;'>
                    Map extracted ligands to protein targets and explore functional 
                    gene-disease associations and UniProt annotations.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Launch Disease Analysis", key="btn_disease", use_container_width=True):
            st.session_state['nav_page'] = "🏥 Disease Enrichment"
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ============================================================================
    # TABBED METADATA SECTIONS - Clean organization
    # ============================================================================
    tab1, tab2, tab3 = st.tabs(["Overview & Features", "Citation Guide", "Development Team"])
    
    # ---- TAB 1: Overview & Features ----
    with tab1:
        st.markdown("""
            <h3 style='color: #2E7D32; background: transparent; margin-top: 1rem;'>
                About TrackMyPDB
            </h3>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        TrackMyPDB is a comprehensive bioinformatics pipeline designed for protein structure 
        analysis, heteroatom extraction, and molecular similarity screening. The platform combines 
        multiple data sources (RCSB PDB, PDBe, PubChem) with robust cheminformatics algorithms 
        to enable target identification and ligand discovery.
        """)
        
        st.markdown("""
            <h4 style='color: #2E7D32; background: transparent; margin-top: 1.5rem;'>
                Key Pipeline Workflows
            </h4>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Heteroatom Extraction Tool**
            - Extract ALL heteroatoms from PDB structures
            - Multi-source data fetching (RCSB PDB, PubChem)
            - Real-time progress tracking
            - Comprehensive ligand database with SMILES
            - Robust error handling and retry logic
            """)
            
            st.markdown("""
            **Molecular Similarity Analyzer**
            - Morgan fingerprint generation
            - Tanimoto similarity scoring
            - Interactive visualizations
            - Statistical analysis reports
            - Ranked similarity results
            """)
        
        with col2:
            st.markdown("""
            **SMILES Database Search**
            - Fast pre-built database queries
            - No UniProt ID required
            - Direct PDB ligand matching
            - Top-N results ranking
            - Co-crystallized ligand identification
            """)
            
            st.markdown("""
            **Disease Enrichment Analysis**
            - Protein-ligand target mapping
            - Gene-disease association analysis
            - UniProt functional annotations
            - Disease category enrichment
            - Interactive filtering and export
            """)
    
    # ---- TAB 2: Citation Guide ----
    with tab2:
        st.markdown("""
            <h3 style='color: #2E7D32; background: transparent; margin-top: 1rem;'>
                📚 Citation Required
            </h3>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        If you use **TrackMyPDB** in your research, publications, or projects, please cite 
        our work using one of the formats below. Proper attribution helps support continued 
        development and improvements.
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # APA Format
        st.markdown("**APA Format:**")
        st.code("""Sharif, S., Gamage, A., Kotawalagedara, K., Sha, S., & Bodun, D. (2025). TrackMyPDB: A comprehensive bioinformatics pipeline for extracting heteroatoms from protein structures and finding molecularly similar compounds using fingerprint-based similarity analysis (Version 2.0) [Computer software]. Standard Seed Corporation. https://trackmypdbsscai.streamlit.app/""", 
                language="text")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # BibTeX Format
        st.markdown("**BibTeX Format:**")
        st.code("""@software{trackmypdb2025,
  author = {Sharif, Suliman and Gamage, Anu and Kotawalagedara, Kalana and Sha, Sakeer and Bodun, Damilola},
  title = {TrackMyPDB: A Comprehensive Bioinformatics Pipeline for Heteroatom Extraction and Molecular Similarity Analysis},
  year = {2025},
  version = {2.0},
  organization = {Standard Seed Corporation},
  url = {https://trackmypdbsscai.streamlit.app/}
}""", language="bibtex")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Acknowledgments
        st.markdown("**Please also acknowledge:**")
        st.markdown("""
        - RCSB PDB (Protein Data Bank)
        - PDBe (Protein Data Bank in Europe)
        - PubChem (National Center for Biotechnology Information)
        - RDKit Cheminformatics Toolkit
        """)
    
    # ---- TAB 3: Development Team ----
    with tab3:
        st.markdown("""
            <h3 style='color: #2E7D32; background: transparent; margin-top: 1rem;'>
                Development Team
            </h3>
        """, unsafe_allow_html=True)
        
        # Corporate backing
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: #f8f9fa; 
                    border-radius: 8px; margin: 1rem 0;'>
            <h4 style='color: #2E7D32; margin-top: 0; background: transparent;'>
                Developed and released by Standard Seed Corporation
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Display SSC logo if available
        if os.path.exists("ssc.png"):
            col_logo1, col_logo2, col_logo3 = st.columns([1, 1, 1])
            with col_logo2:
                st.image("ssc.png", width=200, caption="Standard Seed Corporation")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Team roles in clean layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Project Supervisor**
            - Suliman Sharif
            
            **Lead Engineer**
            - Anu Gamage
            """)
        
        with col2:
            st.markdown("""
            **Associate Engineers**
            - Kalana Kotawalagedara
            - Sakeer Sha
            - Damilola Bodun
            """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.caption("Licensed under MIT License - Open Source Project")

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
    
    # Protein Information Enrichment Section (outside analysis execution)
    # This section persists across reruns using session state
    if 'similarity_results' in st.session_state and not st.session_state['similarity_results'].empty:
        final_results = st.session_state['similarity_results']
        
        # Display results section
        st.markdown("---")
        st.markdown('<div class="section-header">🧬 Protein Target Information</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Fetch additional protein information for the PDB structures found in your similarity analysis results.
        This will query the RCSB PDB database to retrieve:
        - **UniProt IDs** associated with each PDB structure
        - **Protein names/descriptions** for each target
        """)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            fetch_button = st.button("🔍 Fetch Protein Information", type="primary", key="fetch_protein_similarity")
        
        with col2:
            if 'enriched_similarity_results' in st.session_state:
                if st.button("🗑️ Clear Protein Info", key="clear_protein_similarity"):
                    if 'enriched_similarity_results' in st.session_state:
                        del st.session_state['enriched_similarity_results']
                    st.rerun()
        
        # Fetch protein information if button was clicked
        if fetch_button:
            st.info("🔄 Fetching protein information from RCSB PDB...")
            enriched_df = enrich_results_with_protein_info(final_results.copy())
            st.session_state['enriched_similarity_results'] = enriched_df
        
        # Display enriched results if available
        if 'enriched_similarity_results' in st.session_state:
            enriched_df = st.session_state['enriched_similarity_results']
            
            st.subheader("📋 Enriched Results with Protein Information")
            
            # Display enriched table
            display_enriched = enriched_df.copy()
            display_enriched['Tanimoto_Similarity'] = display_enriched['Tanimoto_Similarity'].round(4)
            
            st.dataframe(
                display_enriched,
                use_container_width=True,
                height=min(400 + (len(display_enriched) * 35), 800),
                column_config={
                    "PDB_ID": "PDB ID",
                    "Heteroatom_Code": "Ligand",
                    "Chemical_Name": "Ligand Name",
                    "SMILES": st.column_config.TextColumn("SMILES", width="small"),
                    "Tanimoto_Similarity": st.column_config.NumberColumn(
                        "Similarity",
                        format="%.4f"
                    ),
                    "Formula": "Formula",
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
            
            # Enhanced download section with enriched data
            st.markdown("---")
            st.subheader("📥 Download Enriched Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Full enriched results
                csv_data = enriched_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Enriched Results (CSV)",
                    data=csv_data,
                    file_name=f"TrackMyPDB_similarity_enriched_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download all similarity results with protein information"
                )
            
            with col2:
                # PDB-Protein mapping
                pdb_protein_df = enriched_df[['PDB_ID', 'UniProt_IDs', 'Protein_Names', 'Tanimoto_Similarity']].drop_duplicates()
                pdb_protein_csv = pdb_protein_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download PDB-Protein Mapping (CSV)",
                    data=pdb_protein_csv,
                    file_name=f"TrackMyPDB_PDB_Protein_Mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download PDB IDs with associated UniProt IDs and protein names"
                )

def show_smiles_database_search():
    """
    SMILES Database Search - Query pre-built PDB ligand database using molecular fingerprints
    """
    
    # Clean header - no grey background
    st.markdown("""
        <h2 style='color: #2E7D32; font-weight: 600; margin-bottom: 1.5rem; background: transparent;'>
            SMILES Database Search
        </h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Search the PDB ligands database using your SMILES structure to find similar co-crystallized ligands.
    """)
    
    # Check if database exists
    db_path = "pdb_ligands_trackmypdb_open_source.csv"
    if not os.path.exists(db_path):
        st.error(f"❌ Database file '{db_path}' not found. Please ensure the file exists in the application directory.")
        return
    
    # Load database info for sidebar metrics
    try:
        db_df = pd.read_csv(db_path)
        db_df_valid = db_df[db_df['SMILES'].notna() & (db_df['SMILES'] != '')]
        db_total = len(db_df)
        db_valid = len(db_df_valid)
        db_unique_pdbs = db_df['PDB_ID'].nunique()
        db_unique_ligands = db_df['Heteroatom_Code'].nunique()
    except Exception as e:
        st.error(f"Error loading database: {str(e)}")
        return
    
    # Input section - FULL WIDTH (no columns)
    st.markdown("### Input SMILES")
    
    target_smiles = st.text_area(
        "Target SMILES Structure",
        placeholder="Enter SMILES string (e.g., CCO for ethanol)\nYou can enter multiple SMILES, one per line",
        height=150,
        help="Enter the SMILES representation of your target molecule(s)",
        label_visibility="collapsed"
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
    
    # Advanced parameters in collapsible expander
    with st.expander("⚙️ Advanced Search & Fingerprint Parameters", expanded=False):
        st.markdown("**Search Configuration**")
        col1, col2 = st.columns(2)
        
        with col1:
            top_n = st.slider(
                "Number of Top Results", 
                5, 100, 20, 
                help="Number of top matching PDB IDs to return"
            )
        
        with col2:
            min_similarity = st.slider(
                "Min Similarity Threshold", 
                0.0, 1.0, 0.3, 0.05, 
                help="Minimum Tanimoto similarity score (0-1)"
            )
        
        st.markdown("**Fingerprint Settings**")
        col3, col4 = st.columns(2)
        
        with col3:
            radius = st.selectbox(
                "Morgan Radius", 
                [1, 2, 3], 
                index=1, 
                help="Radius for Morgan fingerprint generation"
            )
        
        with col4:
            n_bits = st.selectbox(
                "Fingerprint Bits", 
                [1024, 2048, 4096], 
                index=1, 
                help="Number of bits in fingerprint"
            )
    
    # Set defaults if expander is not used
    if 'top_n' not in locals():
        top_n = 20
    if 'min_similarity' not in locals():
        min_similarity = 0.3
    if 'radius' not in locals():
        radius = 2
    if 'n_bits' not in locals():
        n_bits = 2048
    
    # Database info metrics - moved to bottom as caption
    st.caption(f"📚 Database: {db_total:,} total ligands | {db_valid:,} valid SMILES | {db_unique_pdbs:,} unique PDB IDs | {db_unique_ligands:,} unique ligand types")
    
    st.markdown("---")
    
    # Run search
    if st.button("🔍 Search Database", type="primary"):
        if not target_smiles:
            st.error("Please enter at least one target SMILES structure")
            return
        
        # Clear any existing enriched results from previous searches
        if 'enriched_results' in st.session_state:
            del st.session_state['enriched_results']
        if 'enriched_top_n' in st.session_state:
            del st.session_state['enriched_top_n']
        
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
                
                # Store ALL results in session state (don't limit here)
                st.session_state['smiles_search_results'] = final_results
                
                st.success(f"🎉 Search completed! Found {len(final_results)} total matches. Scroll down to view results.")
            
            else:
                st.error("No matches found for any of the input SMILES structures")
                
        except Exception as e:
            st.error(f"Error during search: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    # Protein Information Enrichment Section (outside search execution)
    # This section persists across reruns using session state
    if 'smiles_search_results' in st.session_state and len(st.session_state['smiles_search_results']) > 0:
        # Get stored results and apply current top_n limit dynamically
        all_stored_results = st.session_state['smiles_search_results']
        final_results = all_stored_results.head(top_n)
        
        # Debug info
        st.caption(f"🔍 Debug: Slider value = {top_n}, Total stored = {len(all_stored_results)}, Displaying = {len(final_results)}")
        
        # Update the displayed results count message
        if len(all_stored_results) > top_n:
            st.info(f"ℹ️ Showing top {top_n} results out of {len(all_stored_results)} total matches. Adjust the 'Number of Top Results' slider and the display will update automatically.")
        
        # Display results section
        st.markdown("""
            <h3 style='color: #2E7D32; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem; background: transparent;'>
                Search Results
            </h3>
        """, unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Showing", len(final_results))
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
            height=min(400 + (len(display_df) * 35), 800),  # Dynamic height based on number of rows
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
        
        st.markdown("---")
        st.markdown("""
            <h3 style='color: #2E7D32; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem; background: transparent;'>
                Protein Target Information
            </h3>
        """, unsafe_allow_html=True)
        
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
            st.info("🔄 Fetching protein information from RCSB PDB...")
            enriched_df = enrich_results_with_protein_info(final_results.copy())
            st.session_state['enriched_results'] = enriched_df
            st.session_state['enriched_top_n'] = top_n  # Store the top_n value used for enrichment
        
        # Display enriched results if available
        if 'enriched_results' in st.session_state:
            enriched_df = st.session_state['enriched_results']
            
            # Check if top_n has changed since enrichment
            if st.session_state.get('enriched_top_n', top_n) != top_n:
                st.warning(f"⚠️ Note: Protein information was fetched for top {st.session_state.get('enriched_top_n', 'N')} results, but you're now viewing top {top_n}. Click 'Fetch Protein Information' again to update.")
                # Apply current top_n to enriched results
                enriched_df = enriched_df.head(top_n)
            
            st.subheader("📋 Enriched Results with Protein Information")
            
            # Display enriched table
            display_enriched = enriched_df.copy()
            display_enriched['Tanimoto_Similarity'] = display_enriched['Tanimoto_Similarity'].round(4)
            
            st.dataframe(
                display_enriched,
                use_container_width=True,
                height=min(400 + (len(display_enriched) * 35), 800),  # Dynamic height based on number of rows
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
            st.markdown("""
                <h3 style='color: #2E7D32; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem; background: transparent;'>
                    Similarity Distribution
                </h3>
            """, unsafe_allow_html=True)
            
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
    """
    Legacy Search - Complete automated pipeline from UniProt to similarity results
    """
    
    # Clean header - no grey background
    st.markdown("""
        <h2 style='color: #2E7D32; font-weight: 600; margin-bottom: 1.5rem; background: transparent;'>
            Legacy Search
        </h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Automated pipeline: Extract heteroatoms from UniProt proteins → Analyze molecular similarity to target compound.
    """)
    
    # Input section - STACKED VERTICALLY for better space
    st.markdown("### Input Parameters")
    
    # UniProt IDs - full width
    st.markdown("**UniProt IDs**")
    uniprot_input = st.text_area(
        "Enter UniProt IDs (comma-separated or one per line)",
        placeholder="Example: Q9UNQ0, P37231, P06276",
        height=120,
        label_visibility="collapsed"
    )
    
    # Target SMILES - full width
    st.markdown("**Target Molecule SMILES**")
    target_smiles = st.text_area(
        "Enter target SMILES structure",
        placeholder="Example: CCO (ethanol), CC(=O)Oc1ccccc1C(=O)O (aspirin)",
        height=80,
        label_visibility="collapsed"
    )
    
    # Analysis Parameters
    st.markdown("---")
    st.markdown("### Analysis Parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        top_n = st.slider("Top Results", 10, 100, 50)
    with col2:
        min_similarity = st.slider("Min Similarity", 0.0, 1.0, 0.2, 0.1)
    with col3:
        radius = st.selectbox("Morgan Radius", [1, 2, 3], index=1)
    with col4:
        # NEW: Fingerprint Bits configuration
        fingerprint_bits = st.selectbox(
            "Fingerprint Bits", 
            [1024, 2048], 
            index=1,  # Default to 2048
            help="Number of bits for Morgan fingerprint (higher = more precise)"
        )
    
    # Run pipeline button
    st.markdown("---")
    if st.button("🚀 Run Legacy Search", type="primary", use_container_width=True):
        if not uniprot_input or not target_smiles:
            st.error("❌ Please provide both UniProt IDs and target SMILES")
            return
        
        # Parse UniProt IDs
        uniprot_ids = []
        for line in uniprot_input.strip().split('\n'):
            for up_id in line.split(','):
                up_id = up_id.strip()
                if up_id:
                    uniprot_ids.append(up_id)
        
        st.info(f"🔍 Processing {len(uniprot_ids)} UniProt ID(s)...")
        
        # Step 1: Heteroatom Extraction
        st.markdown("### Step 1: Extracting Heteroatoms")
        extractor = HeteroatomExtractor()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress, message):
            progress_bar.progress(progress * 0.7)  # Use 70% for extraction
            status_text.text(f"Extraction: {message}")
        
        try:
            heteroatom_df = extractor.extract_heteroatoms(uniprot_ids, progress_callback=update_progress)
            
            st.success(f"✅ Extracted {len(heteroatom_df)} heteroatom records from {heteroatom_df['PDB_ID'].nunique()} PDB structures")
            
            # Step 2: Similarity Analysis with FIXED fingerprint matching
            st.markdown("### Step 2: Analyzing Molecular Similarity")
            status_text.text("Computing fingerprints and similarity scores...")
            progress_bar.progress(0.7)
            
            # Initialize analyzer with user-specified parameters
            analyzer = MolecularSimilarityAnalyzer(
                radius=radius,
                n_bits=fingerprint_bits  # Use selected fingerprint bits
            )
            
            # Run similarity analysis with improved canonicalization
            similarity_results = analyzer.analyze_similarity(
                target_smiles=target_smiles,
                heteroatom_df=heteroatom_df,
                top_n=top_n,
                min_similarity=min_similarity
            )
            
            progress_bar.progress(1.0)
            status_text.empty()
            progress_bar.empty()
            
            # Store results in session state
            st.session_state['pipeline_heteroatom_data'] = heteroatom_df
            st.session_state['pipeline_similarity_results'] = similarity_results
            
            # Save results
            heteroatom_df.to_csv("complete_pipeline_heteroatoms.csv", index=False)
            if not similarity_results.empty:
                similarity_results.to_csv("complete_pipeline_similarity.csv", index=False)
            
            # Display summary
            if not similarity_results.empty:
                st.success(f"✅ Found {len(similarity_results)} similar molecules above threshold {min_similarity}")
            else:
                st.warning("⚠️ No molecules found above the similarity threshold")
            
            # Download buttons
            st.markdown("---")
            st.markdown("### Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                csv1 = heteroatom_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Heteroatom Results",
                    data=csv1,
                    file_name=f"heteroatoms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                if not similarity_results.empty:
                    csv2 = similarity_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Similarity Results",
                        data=csv2,
                        file_name=f"similarity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
        except Exception as e:
            st.error(f"❌ Pipeline error: {str(e)}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")
            progress_bar.empty()
            status_text.empty()
    
    # Protein Information Enrichment Section (outside pipeline execution)
    # This section persists across reruns using session state
    if 'pipeline_similarity_results' in st.session_state and not st.session_state['pipeline_similarity_results'].empty:
        final_results = st.session_state['pipeline_similarity_results']
        
        # Display results section
        st.markdown("---")
        st.markdown('<div class="section-header">🧬 Protein Target Information</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Fetch additional protein information for the PDB structures found in your pipeline results.
        This will query the RCSB PDB database to retrieve:
        - **UniProt IDs** associated with each PDB structure
        - **Protein names/descriptions** for each target
        """)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            fetch_button = st.button("🔍 Fetch Protein Information", type="primary", key="fetch_protein_pipeline")
        
        with col2:
            if 'enriched_pipeline_results' in st.session_state:
                if st.button("🗑️ Clear Protein Info", key="clear_protein_pipeline"):
                    if 'enriched_pipeline_results' in st.session_state:
                        del st.session_state['enriched_pipeline_results']
                    st.rerun()
        
        # Fetch protein information if button was clicked
        if fetch_button:
            st.info("🔄 Fetching protein information from RCSB PDB...")
            enriched_df = enrich_results_with_protein_info(final_results.copy())
            st.session_state['enriched_pipeline_results'] = enriched_df
        
        # Display enriched results if available
        if 'enriched_pipeline_results' in st.session_state:
            enriched_df = st.session_state['enriched_pipeline_results']
            
            st.subheader("📋 Enriched Pipeline Results with Protein Information")
            
            # Display enriched table
            display_enriched = enriched_df.copy()
            display_enriched['Tanimoto_Similarity'] = display_enriched['Tanimoto_Similarity'].round(4)
            
            st.dataframe(
                display_enriched,
                use_container_width=True,
                height=min(400 + (len(display_enriched) * 35), 800),
                column_config={
                    "PDB_ID": "PDB ID",
                    "Heteroatom_Code": "Ligand",
                    "Chemical_Name": "Ligand Name",
                    "SMILES": st.column_config.TextColumn("SMILES", width="small"),
                    "Tanimoto_Similarity": st.column_config.NumberColumn(
                        "Similarity",
                        format="%.4f"
                    ),
                    "Formula": "Formula",
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
            
            # Enhanced download section with enriched data
            st.markdown("---")
            st.subheader("📥 Download Enriched Pipeline Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Full enriched results
                csv_data = enriched_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Enriched Similarity Results (CSV)",
                    data=csv_data,
                    file_name=f"TrackMyPDB_pipeline_enriched_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download all pipeline results with protein information"
                )
            
            with col2:
                # PDB-Protein mapping
                pdb_protein_df = enriched_df[['PDB_ID', 'UniProt_IDs', 'Protein_Names', 'Tanimoto_Similarity']].drop_duplicates()
                pdb_protein_csv = pdb_protein_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download PDB-Protein Mapping (CSV)",
                    data=pdb_protein_csv,
                    file_name=f"TrackMyPDB_Pipeline_PDB_Protein_Mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download PDB IDs with associated UniProt IDs and protein names"
                )

def show_molecule_visualizer_page():
    """
    Molecule Visualizer & Properties - Simplified UI with 5 Core Properties
    Calculates: MW, LogP, HBD, HBA, TPSA
    """
    
    # Clean header - no grey background, no features list
    st.markdown("""
        <h2 style='color: #2E7D32; font-weight: 600; margin-bottom: 1.5rem; background: transparent;'>
            Molecule Visualizer & Properties
        </h2>
    """, unsafe_allow_html=True)
    
    if not RDKIT_AVAILABLE:
        st.error("❌ RDKit is required for molecular visualization. Please install: `pip install rdkit`")
        return
    
    visualizer = MoleculeVisualizer()
    drawing_tool = ChemicalDrawingTool()
    
    # Input section (now simplified - no "Search by Name")
    smiles = drawing_tool.simple_smiles_input()
    
    if smiles:
        st.markdown("---")
        
        # Layout: 2D Structure + Properties side by side
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 2D Structure")
            visualizer.display_molecule(smiles, caption=f"SMILES: {smiles}")
            
            # SMILES display
            st.code(smiles, language="text")
            
            # Download options
            img = visualizer.smiles_to_image(smiles, size=(800, 800))
            if img:
                buf = BytesIO()
                img.save(buf, format="PNG")
                st.download_button(
                    label="📥 Download Structure (PNG)",
                    data=buf.getvalue(),
                    file_name=f"molecule_{smiles[:20]}.png",
                    mime="image/png"
                )
        
        with col2:
            st.markdown("### Physicochemical Properties")
            
            # Calculate properties (now returns only 5 core descriptors)
            properties = visualizer.calculate_properties(smiles)
            
            if properties:
                # Display 5 core properties in a clean row
                st.markdown("#### Core Descriptors")
                
                cols = st.columns(5)
                
                with cols[0]:
                    st.metric(
                        "MW", 
                        f"{properties['Molecular_Weight']}",
                        help="Molecular Weight (Da)"
                    )
                
                with cols[1]:
                    st.metric(
                        "LogP", 
                        f"{properties['LogP']}",
                        help="Partition coefficient (lipophilicity)"
                    )
                
                with cols[2]:
                    st.metric(
                        "HBD", 
                        f"{properties['HBD']}",
                        help="Hydrogen Bond Donors"
                    )
                
                with cols[3]:
                    st.metric(
                        "HBA", 
                        f"{properties['HBA']}",
                        help="Hydrogen Bond Acceptors"
                    )
                
                with cols[4]:
                    st.metric(
                        "TPSA", 
                        f"{properties['TPSA']}",
                        help="Topological Polar Surface Area (Ų)"
                    )
                
                # Download properties
                st.markdown("---")
                props_df = pd.DataFrame([properties])
                csv = props_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Properties (CSV)",
                    data=csv,
                    file_name=f"properties_{smiles[:20]}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.error("❌ Could not calculate properties for this molecule")


def show_disease_enrichment_page():
    """
    Disease Enrichment Analysis - Map proteins to disease associations and annotations
    """
    
    # Clean header - no grey background
    st.markdown("""
        <h2 style='color: #2E7D32; font-weight: 600; margin-bottom: 1.5rem; background: transparent;'>
            Disease Enrichment Analysis
        </h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Map protein targets to disease associations and filter results by disease categories.
    """)
    
    annotator = DiseaseAnnotator()
    
    # Results source selection
    st.markdown("### Data Source Selection")
    results_source = st.selectbox(
        "Select Results Source",
        ["Similarity Analysis Results", "SMILES Database Search", "Legacy Search", "Manual UniProt Input"],
        help="Choose where to load protein data from"
    )
    
    results_df = None
    
    # Handle different data sources
    if results_source == "Similarity Analysis Results":
        if 'enriched_similarity_results' in st.session_state:
            results_df = st.session_state['enriched_similarity_results']
            st.success(f"✅ Loaded {len(results_df)} similarity analysis results")
        else:
            st.warning("⚠️ No similarity analysis results found. Please run a similarity analysis first.")
    
    elif results_source == "SMILES Database Search":
        if 'enriched_results' in st.session_state:
            results_df = st.session_state['enriched_results']
            st.success(f"✅ Loaded {len(results_df)} SMILES search results")
        else:
            st.warning("⚠️ No SMILES search results found. Please run a database search first.")
    
    elif results_source == "Legacy Search":
        if 'enriched_pipeline_results' in st.session_state:
            results_df = st.session_state['enriched_pipeline_results']
            st.success(f"✅ Loaded {len(results_df)} legacy search results")
        else:
            st.warning("⚠️ No legacy search results found. Please run the legacy pipeline first.")
    
    elif results_source == "Manual UniProt Input":
        # ALWAYS show the input box when Manual is selected
        st.markdown("**Enter UniProt IDs**")
        uniprot_input = st.text_area(
            "UniProt IDs (comma, space, or newline separated)",
            placeholder="Examples:\nP04637, Q9UNQ0, P37231\nor\nP04637 Q9UNQ0 P37231\nor\nP04637\nQ9UNQ0\nP37231",
            height=120,
            help="Enter UniProt IDs separated by commas, spaces, or newlines",
            label_visibility="collapsed"
        )
        
        if uniprot_input:
            if st.button("🔍 Create Dataset from UniProt IDs", type="primary"):
                # ROBUST PARSING: Handle commas, spaces, newlines, and mixed delimiters
                import re
                # Split by any combination of whitespace (spaces, tabs, newlines) and/or commas
                uniprot_ids = [uid.strip().upper() for uid in re.split(r'[\s,]+', uniprot_input) if uid.strip()]
                
                if uniprot_ids:
                    # Create a simple DataFrame
                    results_df = pd.DataFrame({
                        'UniProt_IDs': uniprot_ids,
                        'PDB_ID': ['N/A'] * len(uniprot_ids)
                    })
                    # Store in session state so it persists
                    st.session_state['manual_uniprot_df'] = results_df
                    st.success(f"✅ Parsed {len(uniprot_ids)} UniProt ID(s): {', '.join(uniprot_ids[:5])}{' ...' if len(uniprot_ids) > 5 else ''}")
                else:
                    st.error("❌ No valid UniProt IDs found. Please check your input.")
        else:
            st.info("ℹ️ Enter UniProt IDs above and click the button to proceed.")
        
        # Load from session state if exists
        if 'manual_uniprot_df' in st.session_state:
            results_df = st.session_state['manual_uniprot_df']
    
    # Main analysis section
    if results_df is not None and len(results_df) > 0:
        
        # Disease filter options
        st.markdown("---")
        st.markdown("### Disease Filter Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            disease_category = st.multiselect(
                "Select Disease Categories",
                list(DISEASE_CATEGORIES.keys()),
                help="Pre-defined disease categories"
            )
        
        with col2:
            custom_keywords = st.text_input(
                "Custom Disease Keywords (comma-separated)",
                placeholder="leukemia, breast cancer, COVID-19",
                help="Add your own disease keywords"
            )
        
        # Run enrichment
        st.markdown("---")
        if st.button("🚀 Run Disease Enrichment", type="primary", use_container_width=True):
            with st.spinner("Fetching disease annotations from UniProt..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                enriched_df = annotator.enrich_results_with_diseases(
                    results_df.copy(),
                    progress_callback=update_progress
                )
                
                progress_bar.empty()
                status_text.empty()
                
                st.session_state['disease_enriched_results'] = enriched_df
                st.success("✅ Disease enrichment completed!")
        
        # Display enriched results
        if 'disease_enriched_results' in st.session_state:
            enriched_df = st.session_state['disease_enriched_results']
            
            st.markdown("---")
            st.markdown("""
                <h3 style='color: #2E7D32; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; background: transparent;'>
                    Disease-Enriched Results
                </h3>
            """, unsafe_allow_html=True)
            
            # Apply filters if selected
            filtered_df = enriched_df.copy()
            
            if disease_category or custom_keywords:
                keywords = []
                
                # Add category keywords
                for cat in disease_category:
                    keywords.extend(DISEASE_CATEGORIES[cat])
                
                # Add custom keywords
                if custom_keywords:
                    keywords.extend([k.strip() for k in custom_keywords.split(',')])
                
                # Filter
                mask = filtered_df['Disease_Associations'].apply(
                    lambda x: any(kw.lower() in str(x).lower() for kw in keywords) if x != 'N/A' else False
                )
                filtered_df = filtered_df[mask]
                
                st.info(f"🔍 Filtered to {len(filtered_df)} results matching disease criteria")
            
            # Display results
            st.dataframe(filtered_df, use_container_width=True, height=400)
            
            # Disease summary
            st.markdown("---")
            st.markdown("""
                <h3 style='color: #2E7D32; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; background: transparent;'>
                    Disease Summary
                </h3>
            """, unsafe_allow_html=True)
            
            # Count disease mentions
            all_diseases = []
            for disease_str in filtered_df['Disease_Associations']:
                if disease_str and disease_str != 'N/A' and disease_str != 'No disease associations':
                    all_diseases.extend([d.strip() for d in str(disease_str).split('|')])
            
            if all_diseases:
                disease_counts = pd.Series(all_diseases).value_counts().head(20)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    import plotly.express as px
                    fig = px.bar(
                        x=disease_counts.values,
                        y=disease_counts.index,
                        orientation='h',
                        title='Top 20 Disease Associations',
                        labels={'x': 'Count', 'y': 'Disease'},
                        color=disease_counts.values,
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Statistics**")
                    st.metric("Total Diseases", len(all_diseases))
                    st.metric("Unique Diseases", len(set(all_diseases)))
                    st.metric("Most Common", disease_counts.index[0] if len(disease_counts) > 0 else "N/A")
                    st.metric("Frequency", disease_counts.values[0] if len(disease_counts) > 0 else 0)
            else:
                st.info("No disease associations found in filtered results")
            
            # Download
            st.markdown("---")
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Disease-Enriched Results (CSV)",
                data=csv_data,
                file_name=f"disease_enriched_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Data sources info - moved to bottom expander
    st.markdown("---")
    with st.expander("ℹ️ About Disease Mapping & Data Sources"):
        st.markdown("""
        ### Data Sources
        
        This tool retrieves disease associations from:
        
        - **UniProt Disease Annotations**: Curated disease involvement information from the UniProt database
        - **Gene-Disease Associations**: Functional links between genes and disease phenotypes
        - **Clinical Relevance Filtering**: Pre-defined disease categories for targeted analysis
        
        ### Disease Categories Available
        
        The tool includes pre-defined categories covering major disease areas:
        - Cancer, Diabetes, Cardiovascular diseases, Neurological disorders, Infectious diseases, and more
        
        ### How It Works
        
        1. Select a data source (previous analysis results or manual UniProt IDs)
        2. Optionally filter by disease categories or custom keywords
        3. Run enrichment to fetch disease annotations from UniProt
        4. View results with disease associations mapped to each protein
        5. Download enriched data for further analysis
        """)

if __name__ == "__main__":
    main()
