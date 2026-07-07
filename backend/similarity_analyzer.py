"""
TrackMyPDB Molecular Similarity Analyzer (Corrected Algorithm)
@author Anu Gamage

Based on the proven Colab notebook algorithm for accurate molecular similarity analysis.
Licensed under MIT License - Open Source Project
"""

# Force matplotlib to use non-GUI backend (fixes libXrender.so.1 error on Linux/WSL)
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs
from rdkit import DataStructs
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import warnings

warnings.filterwarnings('ignore')


class MolecularSimilarityAnalyzer:
    """
    A comprehensive tool for molecular similarity analysis using Morgan fingerprints
    and Tanimoto similarity scoring - corrected algorithm from Colab notebook.
    """

    def __init__(self, radius=2, n_bits=2048):
        self.radius = radius
        self.n_bits = n_bits
        self.fingerprints = {}
        self.valid_molecules = {}

    def smiles_to_fingerprint(self, smiles):
        """
        Convert SMILES string to Morgan fingerprint.
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            Morgan fingerprint or None if invalid
        """
        try:
            if not smiles or pd.isna(smiles) or smiles.strip() == '':
                return None

            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return None

            # Generate Morgan fingerprint - exact same as Colab
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, self.radius, nBits=self.n_bits
            )
            return fp
        except Exception as e:
            st.warning(f"⚠️ Error processing SMILES '{smiles}': {e}")
            return None

    def calculate_tanimoto_similarity(self, fp1, fp2):
        """
        Calculate Tanimoto similarity between two fingerprints.
        
        Args:
            fp1, fp2: RDKit fingerprint objects
            
        Returns:
            float: Tanimoto similarity score (0-1)
        """
        if fp1 is None or fp2 is None:
            return 0.0

        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def load_and_process_dataframe(self, df, smiles_column='SMILES',
                                  pdb_column='PDB_ID',
                                  ligand_column='Heteroatom_Code'):
        """
        Load DataFrame and process SMILES to fingerprints.
        Uses the exact same logic as the Colab notebook.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            smiles_column (str): Name of SMILES column
            pdb_column (str): Name of PDB ID column
            ligand_column (str): Name of ligand code column
            
        Returns:
            pd.DataFrame: Processed DataFrame with valid molecules only
        """
        st.info("🔄 Processing DataFrame and computing fingerprints...")

        # Filter out rows without SMILES - exact same as Colab
        valid_df = df[df[smiles_column].notna() & (df[smiles_column] != '')].copy()
        st.info(f"📊 Found {len(valid_df)} entries with SMILES out of {len(df)} total")

        # Compute fingerprints - exact same logic as Colab
        fingerprints = []
        valid_indices = []

        # Progress tracking for Streamlit
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, (original_idx, row) in enumerate(valid_df.iterrows()):
            # Update progress
            progress = (idx + 1) / len(valid_df)
            progress_bar.progress(progress)
            status_text.text(f"Computing fingerprints... {idx+1}/{len(valid_df)}")

            smiles = row[smiles_column]
            fp = self.smiles_to_fingerprint(smiles)

            if fp is not None:
                fingerprints.append(fp)
                valid_indices.append(original_idx)

                # Store for later use - exact same as Colab
                key = f"{row[pdb_column]}_{row[ligand_column]}"
                self.fingerprints[key] = fp
                self.valid_molecules[key] = {
                    'smiles': smiles,
                    'pdb_id': row[pdb_column],
                    'ligand_code': row[ligand_column],
                    'chemical_name': row.get('Chemical_Name', ''),
                    'formula': row.get('Formula', '')
                }

        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()

        # Create processed DataFrame - exact same as Colab
        processed_df = valid_df.loc[valid_indices].copy()
        processed_df['Fingerprint'] = fingerprints

        st.success(f"✅ Successfully processed {len(processed_df)} molecules with valid fingerprints")
        return processed_df

    def find_similar_ligands(self, target_smiles, processed_df, top_n=50, min_similarity=0.0):
        """
        Find ligands similar to target molecule.
        Uses the exact same logic as the Colab notebook.
        
        Args:
            target_smiles (str): Target molecule SMILES
            processed_df (pd.DataFrame): DataFrame with computed fingerprints
            top_n (int): Number of top results to return
            min_similarity (float): Minimum similarity threshold
            
        Returns:
            pd.DataFrame: Sorted DataFrame with similarity scores
        """
        st.info(f"🎯 Analyzing similarity to target: {target_smiles}")

        # Generate target fingerprint - exact same as Colab
        target_fp = self.smiles_to_fingerprint(target_smiles)
        if target_fp is None:
            raise ValueError(f"Invalid target SMILES: {target_smiles}")

        # Calculate similarities - exact same as Colab
        similarities = []
        st.info("🔍 Computing similarities...")

        # Progress tracking for Streamlit
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, (original_idx, row) in enumerate(processed_df.iterrows()):
            # Update progress
            progress = (idx + 1) / len(processed_df)
            progress_bar.progress(progress)
            status_text.text(f"Computing similarities... {idx+1}/{len(processed_df)}")

            ligand_fp = row['Fingerprint']
            similarity = self.calculate_tanimoto_similarity(target_fp, ligand_fp)
            similarities.append(similarity)

        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()

        # Add similarity scores to DataFrame - exact same as Colab
        result_df = processed_df.copy()
        result_df['Tanimoto_Similarity'] = similarities

        # Filter by minimum similarity and sort - exact same as Colab
        result_df = result_df[result_df['Tanimoto_Similarity'] >= min_similarity]
        result_df = result_df.sort_values('Tanimoto_Similarity', ascending=False)

        # Return top results - exact same as Colab
        top_results = result_df.head(top_n)

        st.success(f"🏆 Found {len(result_df)} ligands above similarity threshold {min_similarity}")
        st.info(f"📋 Returning top {len(top_results)} results")

        return top_results

    def display_similarity_results(self, target_smiles, similarity_results):
        """
        Display comprehensive similarity analysis results in Streamlit.
        Based on the Colab notebook report logic.
        
        Args:
            target_smiles (str): Target molecule SMILES
            similarity_results (pd.DataFrame): Results from find_similar_ligands
        """
        st.markdown("---")
        st.subheader("🎯 Molecular Similarity Analysis Results")
        
        # Basic metrics - same as Colab report
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🧪 Target SMILES", target_smiles[:30] + "..." if len(target_smiles) > 30 else target_smiles)
        with col2:
            st.metric("📊 Total Similar Ligands", len(similarity_results))
        with col3:
            if len(similarity_results) > 0:
                st.metric("🏆 Highest Similarity", f"{similarity_results['Tanimoto_Similarity'].max():.4f}")

        if len(similarity_results) > 0:
            # Statistics - same as Colab report
            st.subheader("📈 Analysis Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🏆 Highest Score", f"{similarity_results['Tanimoto_Similarity'].max():.4f}")
            with col2:
                st.metric("📈 Average Score", f"{similarity_results['Tanimoto_Similarity'].mean():.4f}")
            with col3:
                st.metric("📉 Lowest Score", f"{similarity_results['Tanimoto_Similarity'].min():.4f}")
            with col4:
                st.metric("🏗️ Unique PDBs", similarity_results['PDB_ID'].nunique())

            # Top 10 results - same as Colab report
            st.subheader("🔝 Top 10 Most Similar Ligands")
            top_10 = similarity_results.head(10)
            
            # Create a clean display table
            display_df = top_10[[
                'PDB_ID', 
                'Heteroatom_Code', 
                'Chemical_Name', 
                'SMILES', 
                'Tanimoto_Similarity', 
                'Formula'
            ]].copy()
            
            # Format similarity scores
            display_df['Tanimoto_Similarity'] = display_df['Tanimoto_Similarity'].round(4)
            
            # Display results table
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "PDB_ID": "PDB ID",
                    "Heteroatom_Code": "Ligand Code", 
                    "Chemical_Name": "Chemical Name",
                    "SMILES": st.column_config.TextColumn("SMILES", width="medium"),
                    "Tanimoto_Similarity": st.column_config.NumberColumn(
                        "Similarity Score",
                        help="Tanimoto similarity (0-1, higher is better)",
                        format="%.4f"
                    ),
                    "Formula": "Molecular Formula"
                },
                hide_index=True
            )

            # Similarity distribution - same as Colab report
            st.subheader("📊 Similarity Distribution")
            bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            hist, _ = np.histogram(similarity_results['Tanimoto_Similarity'], bins=bins)
            
            dist_df = pd.DataFrame({
                'Range': [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)],
                'Count': hist
            })
            
            st.dataframe(dist_df, use_container_width=True)

            # Return formatted results for download
            return display_df
        else:
            st.warning("❌ No similar ligands found above the threshold")
            return pd.DataFrame()

    def create_similarity_visualizations(self, similarity_results, target_smiles):
        """
        Create comprehensive visualizations based on the Colab notebook plots.
        
        Args:
            similarity_results (pd.DataFrame): Results from similarity analysis
            target_smiles (str): Target molecule SMILES
        """
        if len(similarity_results) == 0:
            st.warning("⚠️ No data available for visualization")
            return

        st.subheader("📊 Similarity Analysis Visualizations")

        # Create subplots matching the Colab notebook layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Distribution of Similarity Scores', 
                'Top 20 Most Similar Ligands', 
                'PDB Structures: Ligand Count vs Avg Similarity', 
                'Cumulative Distribution of Similarities'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # 1. Histogram of similarities - same as Colab
        fig.add_trace(
            go.Histogram(
                x=similarity_results['Tanimoto_Similarity'], 
                nbinsx=20, 
                name='Similarity Distribution',
                marker_color='skyblue',
                opacity=0.7
            ),
            row=1, col=1
        )

        # 2. Top 20 ligands bar plot - same as Colab
        top_20 = similarity_results.head(20)
        fig.add_trace(
            go.Bar(
                x=top_20['Tanimoto_Similarity'],
                y=[f"{row['PDB_ID']}-{row['Heteroatom_Code']}" for _, row in top_20.iterrows()],
                orientation='h',
                name='Top 20 Ligands',
                marker_color='lightgreen',
                text=[f"{score:.3f}" for score in top_20['Tanimoto_Similarity']],
                textposition='auto'
            ),
            row=1, col=2
        )

        # 3. Similarity vs PDB count - same as Colab
        pdb_similarity = similarity_results.groupby('PDB_ID')['Tanimoto_Similarity'].agg(['mean', 'count']).reset_index()
        fig.add_trace(
            go.Scatter(
                x=pdb_similarity['count'], 
                y=pdb_similarity['mean'],
                mode='markers',
                marker=dict(
                    size=10, 
                    color=pdb_similarity['mean'], 
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Avg Similarity")
                ),
                text=[f"PDB: {pdb}<br>Ligands: {count}<br>Avg Sim: {avg:.3f}" 
                      for pdb, count, avg in zip(pdb_similarity['PDB_ID'], 
                                                pdb_similarity['count'],
                                                pdb_similarity['mean'])],
                hovertemplate='<b>%{text}</b><extra></extra>',
                name='PDB Analysis'
            ),
            row=2, col=1
        )

        # 4. Cumulative distribution - same as Colab
        sorted_similarities = np.sort(similarity_results['Tanimoto_Similarity'])
        cumulative = np.arange(1, len(sorted_similarities) + 1) / len(sorted_similarities)
        fig.add_trace(
            go.Scatter(
                x=sorted_similarities,
                y=cumulative,
                mode='lines',
                line=dict(color='red', width=2),
                name='Cumulative Distribution'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title_text=f"Molecular Similarity Analysis<br>Target: {target_smiles[:50]}{'...' if len(target_smiles) > 50 else ''}",
            showlegend=False,
            height=800
        )

        # Update axis labels
        fig.update_xaxes(title_text="Tanimoto Similarity", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        
        fig.update_xaxes(title_text="Tanimoto Similarity", row=1, col=2)
        fig.update_yaxes(title_text="Ligand", row=1, col=2)
        
        fig.update_xaxes(title_text="Number of Ligands per PDB", row=2, col=1)
        fig.update_yaxes(title_text="Average Similarity Score", row=2, col=1)
        
        fig.update_xaxes(title_text="Tanimoto Similarity", row=2, col=2)
        fig.update_yaxes(title_text="Cumulative Probability", row=2, col=2)

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

    def create_similarity_report(self, target_smiles, similarity_results):
        """Create comprehensive similarity analysis report"""
        st.subheader("🎯 Molecular Similarity Analysis Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Target SMILES", target_smiles)
            st.metric("Total Similar Ligands", len(similarity_results))
        
        with col2:
            if len(similarity_results) > 0:
                st.metric("Highest Similarity", f"{similarity_results['Tanimoto_Similarity'].max():.4f}")
                st.metric("Average Similarity", f"{similarity_results['Tanimoto_Similarity'].mean():.4f}")

        if len(similarity_results) > 0:
            st.subheader("🔝 Top 10 Most Similar Ligands")
            top_10 = similarity_results.head(10)
            
            display_df = top_10[['PDB_ID', 'Heteroatom_Code', 'Chemical_Name', 'Tanimoto_Similarity', 'SMILES']].copy()
            display_df['Tanimoto_Similarity'] = display_df['Tanimoto_Similarity'].round(4)
            st.dataframe(display_df, use_container_width=True)

    def create_interactive_plots(self, similarity_results, target_smiles):
        """Create interactive visualizations using Plotly"""
        if len(similarity_results) == 0:
            st.warning("No data to plot")
            return

        # Simple histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=similarity_results['Tanimoto_Similarity'], 
                                  nbinsx=20, name='Similarity Distribution'))
        fig.update_layout(title=f"Similarity Distribution for {target_smiles[:30]}...",
                         xaxis_title='Tanimoto Similarity',
                         yaxis_title='Frequency')
        
        st.plotly_chart(fig, use_container_width=True)

    def analyze_similarity(self, target_smiles, heteroatom_df, top_n=50, min_similarity=0.01):
        """
        Complete similarity analysis workflow.
        Uses the exact same parameters and logic as the Colab notebook.
        
        Args:
            target_smiles (str): Target molecule SMILES
            heteroatom_df (pd.DataFrame): DataFrame with heteroatom data
            top_n (int): Number of top results to return (default: 50)
            min_similarity (float): Minimum similarity threshold (default: 0.01)
            
        Returns:
            pd.DataFrame: Similarity results ready for download
        """
        # Process DataFrame
        processed_df = self.load_and_process_dataframe(heteroatom_df)

        if len(processed_df) == 0:
            st.error("❌ No valid SMILES found in the DataFrame!")
            return pd.DataFrame()

        # Find similar ligands
        similarity_results = self.find_similar_ligands(
            target_smiles=target_smiles,
            processed_df=processed_df,
            top_n=top_n,
            min_similarity=min_similarity
        )

        # Display results
        display_df = self.display_similarity_results(target_smiles, similarity_results)

        # Create visualizations
        if len(similarity_results) > 0:
            self.create_similarity_visualizations(similarity_results, target_smiles)

        return display_df 