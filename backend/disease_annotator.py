"""
TrackMyPDB Disease Annotator
@author: Anu Gamage, Standard Seed Corporation

Maps proteins to disease associations for enrichment analysis
Licensed under MIT License - Open Source Project
"""

import requests
import pandas as pd
import streamlit as st
from typing import Dict, List
import time


class DiseaseAnnotator:
    """Fetch and manage disease annotations for proteins"""
    
    def __init__(self):
        self.cache = {}
    
    def fetch_uniprot_disease_annotations(self, uniprot_id):
        """
        Fetch disease annotations from UniProt API
        
        Args:
            uniprot_id (str): UniProt accession
            
        Returns:
            dict: Disease information
        """
        if uniprot_id in self.cache:
            return self.cache[uniprot_id]
        
        try:
            url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                diseases = []
                
                # Extract disease comments
                if 'comments' in data:
                    for comment in data['comments']:
                        if comment.get('commentType') == 'DISEASE':
                            disease_info = {
                                'disease_name': comment.get('disease', {}).get('diseaseId', ''),
                                'description': comment.get('disease', {}).get('description', ''),
                                'acronym': comment.get('disease', {}).get('acronym', ''),
                                'references': comment.get('disease', {}).get('evidences', [])
                            }
                            diseases.append(disease_info)
                
                result = {
                    'uniprot_id': uniprot_id,
                    'diseases': diseases,
                    'protein_name': data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', ''),
                    'gene_names': [g.get('geneName', {}).get('value', '') for g in data.get('genes', [])],
                    'status': 'success'
                }
                
                self.cache[uniprot_id] = result
                return result
            else:
                return {'uniprot_id': uniprot_id, 'diseases': [], 'status': f'http_error_{response.status_code}'}
                
        except Exception as e:
            return {'uniprot_id': uniprot_id, 'diseases': [], 'status': f'error: {str(e)}'}
    
    def search_disease_keywords(self, disease_annotations, keywords):
        """
        Filter disease annotations by keywords
        
        Args:
            disease_annotations (list): List of disease annotation dicts
            keywords (list): Keywords to search (e.g., ['diabetes', 'cancer'])
            
        Returns:
            list: Filtered annotations matching keywords
        """
        if not keywords:
            return disease_annotations
        
        matching = []
        keywords_lower = [k.lower() for k in keywords]
        
        for annot in disease_annotations:
            for disease in annot.get('diseases', []):
                disease_text = (
                    disease.get('disease_name', '') + ' ' + 
                    disease.get('description', '') + ' ' + 
                    disease.get('acronym', '')
                ).lower()
                
                if any(keyword in disease_text for keyword in keywords_lower):
                    matching.append(annot)
                    break
        
        return matching
    
    def enrich_results_with_diseases(self, results_df, progress_callback=None):
        """
        Add disease annotations to results DataFrame
        
        Args:
            results_df (pd.DataFrame): Results with UniProt_IDs column
            progress_callback: Optional progress update function
            
        Returns:
            pd.DataFrame: Enriched with disease columns
        """
        # Extract unique UniProt IDs
        unique_uniprots = set()
        
        if 'UniProt_IDs' in results_df.columns:
            for ids_str in results_df['UniProt_IDs']:
                if ids_str and ids_str != 'N/A':
                    unique_uniprots.update([uid.strip() for uid in str(ids_str).split(',')])
        else:
            st.warning("No UniProt_IDs column found in results. Please fetch protein information first.")
            return results_df
        
        unique_uniprots = list(unique_uniprots)
        
        if not unique_uniprots:
            st.warning("No UniProt IDs found to enrich with disease data")
            results_df['Disease_Associations'] = 'No UniProt IDs available'
            return results_df
        
        # Fetch disease annotations
        disease_map = {}
        
        for idx, uniprot_id in enumerate(unique_uniprots):
            if progress_callback:
                progress_callback((idx + 1) / len(unique_uniprots), 
                                f"Fetching disease info for {uniprot_id}")
            
            disease_info = self.fetch_uniprot_disease_annotations(uniprot_id)
            disease_map[uniprot_id] = disease_info
            
            time.sleep(0.2)  # Respectful API delay
        
        # Add disease columns to DataFrame
        def get_diseases(uniprot_ids_str):
            if not uniprot_ids_str or uniprot_ids_str == 'N/A':
                return 'N/A'
            
            diseases = []
            for uid in str(uniprot_ids_str).split(','):
                uid = uid.strip()
                if uid in disease_map:
                    for disease in disease_map[uid].get('diseases', []):
                        disease_name = disease.get('disease_name', '')
                        if disease_name:
                            diseases.append(disease_name)
            
            return ' | '.join(diseases) if diseases else 'No disease associations'
        
        results_df['Disease_Associations'] = results_df['UniProt_IDs'].apply(get_diseases)
        
        return results_df


# Disease category mappings for filtering
DISEASE_CATEGORIES = {
    'Diabetes': ['diabetes', 'diabetic', 'insulin resistance', 'hyperglycemia', 'type 2 diabetes', 'type 1 diabetes'],
    'Cancer': ['cancer', 'carcinoma', 'tumor', 'oncogene', 'malignancy', 'neoplasm', 'leukemia', 'lymphoma'],
    'Cardiovascular': ['cardiovascular', 'heart', 'cardiac', 'hypertension', 'atherosclerosis', 'coronary'],
    'Neurological': ['alzheimer', 'parkinson', 'dementia', 'neurodegeneration', 'epilepsy', 'multiple sclerosis'],
    'Immune': ['immune', 'autoimmune', 'inflammation', 'arthritis', 'lupus', 'immunodeficiency'],
    'Infectious': ['viral', 'bacterial', 'infection', 'HIV', 'tuberculosis', 'hepatitis', 'COVID'],
    'Metabolic': ['metabolic', 'obesity', 'lipid', 'cholesterol', 'fatty acid', 'dyslipidemia'],
}
