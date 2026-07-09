"""
TrackMyPDB MCP Server
Exposes TrackMyPDB functionality as Model Context Protocol (MCP) tools
Allows Claude to interact with heteroatom extraction and similarity analysis
"""

import json
import sys
import os
from typing import Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add backend to path — try multiple locations so this works whether the
# MCP server sits in the repo root or inside the agent/ subfolder.
_here = os.path.dirname(os.path.abspath(__file__))
_candidate_paths = [
    _here,                                # same folder as this file
    os.path.dirname(_here),               # parent (repo root, if file is in agent/)
    os.path.join(_here, 'TrackMyPDB-Open-Source-sakeer'),  # local dev layout
]
for _p in _candidate_paths:
    if os.path.isdir(os.path.join(_p, 'backend')):
        sys.path.insert(0, _p)
        break

try:
    from backend.heteroatom_extractor import HeteroatomExtractor
    from backend.similarity_analyzer import MolecularSimilarityAnalyzer
except ImportError:
    print("Warning: Could not import TrackMyPDB modules. Some features may be limited.")
    HeteroatomExtractor = None
    MolecularSimilarityAnalyzer = None


class MCPServer:
    """MCP Server for TrackMyPDB"""
    
    def __init__(self):
        self.extractor = HeteroatomExtractor() if HeteroatomExtractor else None
        self.analyzer = MolecularSimilarityAnalyzer() if MolecularSimilarityAnalyzer else None
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def define_tools(self) -> list:
        """Define all available tools for Claude"""
        return [
            {
                "name": "extract_heteroatoms",
                "description": "Extract heteroatoms from PDB structures associated with UniProt proteins. Returns heteroatom information including SMILES, compound names, and chemical data.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "uniprot_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of UniProt protein identifiers (e.g., ['Q9UNQ0', 'P37231'])"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of heteroatoms to extract per protein (default: 1000)",
                            "default": 1000
                        }
                    },
                    "required": ["uniprot_ids"]
                }
            },
            {
                "name": "analyze_molecular_similarity",
                "description": "Find ligands similar to a target molecule using Morgan fingerprints and Tanimoto similarity metric.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "target_smiles": {
                            "type": "string",
                            "description": "SMILES representation of the target molecule (e.g., 'CCO' for ethanol)"
                        },
                        "molecule_list": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of SMILES strings to compare against target molecule"
                        },
                        "morgan_radius": {
                            "type": "integer",
                            "enum": [1, 2, 3],
                            "description": "Morgan fingerprint radius (default: 2)",
                            "default": 2
                        },
                        "fingerprint_bits": {
                            "type": "integer",
                            "enum": [1024, 2048, 4096],
                            "description": "Number of bits in fingerprint (default: 2048)",
                            "default": 2048
                        },
                        "min_similarity": {
                            "type": "number",
                            "description": "Minimum similarity threshold (0.0-1.0, default: 0.2)",
                            "default": 0.2
                        },
                        "top_n": {
                            "type": "integer",
                            "description": "Return top N results (default: 50)",
                            "default": 50
                        }
                    },
                    "required": ["target_smiles", "molecule_list"]
                }
            },
            {
                "name": "get_pdb_structures",
                "description": "Get all available PDB structures for a UniProt protein ID",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "uniprot_id": {
                            "type": "string",
                            "description": "UniProt protein identifier"
                        }
                    },
                    "required": ["uniprot_id"]
                }
            },
            {
                "name": "get_protein_info",
                "description": "Get authoritative protein information (name, gene, organism, function) from the UniProt database for a given UniProt ID. ALWAYS use this before describing what a protein is — do not rely on prior knowledge, as protein IDs are easy to confuse.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "uniprot_id": {
                            "type": "string",
                            "description": "UniProt protein identifier (e.g., P37231, Q9UNQ0)"
                        }
                    },
                    "required": ["uniprot_id"]
                }
            },
            {
                "name": "validate_smiles",
                "description": "Validate if a SMILES string is valid and return molecular properties",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "smiles": {
                            "type": "string",
                            "description": "SMILES string to validate"
                        }
                    },
                    "required": ["smiles"]
                }
            },
            {
                "name": "batch_heteroatom_extraction",
                "description": "Extract heteroatoms from multiple UniProt IDs in a batch operation",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "uniprot_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of UniProt IDs"
                        },
                        "output_format": {
                            "type": "string",
                            "enum": ["json", "csv"],
                            "description": "Output format (default: json)",
                            "default": "json"
                        }
                    },
                    "required": ["uniprot_ids"]
                }
            }
        ]
    
    async def execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        """Execute a tool and return results"""
        
        if tool_name == "extract_heteroatoms":
            return await self._extract_heteroatoms(
                tool_input["uniprot_ids"],
                tool_input.get("max_results", 1000)
            )
        
        elif tool_name == "analyze_molecular_similarity":
            return await self._analyze_similarity(
                tool_input["target_smiles"],
                tool_input["molecule_list"],
                tool_input.get("morgan_radius", 2),
                tool_input.get("fingerprint_bits", 2048),
                tool_input.get("min_similarity", 0.2),
                tool_input.get("top_n", 50)
            )
        
        elif tool_name == "get_pdb_structures":
            return await self._get_pdb_structures(tool_input["uniprot_id"])
        
        elif tool_name == "get_protein_info":
            return await self._get_protein_info(tool_input["uniprot_id"])
        
        elif tool_name == "validate_smiles":
            return await self._validate_smiles(tool_input["smiles"])
        
        elif tool_name == "batch_heteroatom_extraction":
            return await self._batch_extraction(
                tool_input["uniprot_ids"],
                tool_input.get("output_format", "json")
            )
        
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    async def _extract_heteroatoms(self, uniprot_ids: list, max_results: int) -> dict:
        """Extract heteroatoms from UniProt IDs"""
        try:
            if not self.extractor:
                return {"error": "HeteroatomExtractor not available"}
            
            results = []
            for uniprot_id in uniprot_ids:
                # This would need to be adapted to work async
                # For now, return structure
                pdb_ids = self.extractor.get_pdbs_for_uniprot(uniprot_id)
                results.append({
                    "uniprot_id": uniprot_id,
                    "pdb_count": len(pdb_ids),
                    "pdb_ids": pdb_ids[:10],  # Return first 10
                    "status": "ready_for_extraction"
                })
            
            return {
                "status": "success",
                "total_proteins": len(uniprot_ids),
                "results": results,
                "message": "Use extract_heteroatom_details for full extraction"
            }
        
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    async def _analyze_similarity(self, target_smiles: str, molecule_list: list, 
                                 morgan_radius: int, fingerprint_bits: int, 
                                 min_similarity: float, top_n: int) -> dict:
        """Analyze molecular similarity"""
        try:
            if not self.analyzer:
                return {"error": "MolecularSimilarityAnalyzer not available"}
            
            results = self.analyzer.compute_similarity(
                target_smiles=target_smiles,
                smiles_list=molecule_list,
                morgan_radius=morgan_radius,
                fingerprint_bits=fingerprint_bits,
                min_similarity=min_similarity,
                top_n=top_n
            )
            
            return {
                "status": "success",
                "target_smiles": target_smiles,
                "total_compared": len(molecule_list),
                "results": results,
                "parameters": {
                    "morgan_radius": morgan_radius,
                    "fingerprint_bits": fingerprint_bits,
                    "min_similarity": min_similarity
                }
            }
        
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    async def _get_pdb_structures(self, uniprot_id: str) -> dict:
        """Get PDB structures for a UniProt ID"""
        try:
            if not self.extractor:
                return {"error": "HeteroatomExtractor not available"}
            
            pdb_ids = self.extractor.get_pdbs_for_uniprot(uniprot_id)
            
            return {
                "status": "success",
                "uniprot_id": uniprot_id,
                "total_structures": len(pdb_ids),
                "pdb_ids": pdb_ids,
                "message": f"Found {len(pdb_ids)} PDB structures"
            }
        
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    async def _get_protein_info(self, uniprot_id: str) -> dict:
        """Fetch authoritative protein information from the UniProt REST API"""
        try:
            import requests
            
            uniprot_id = uniprot_id.strip().upper()
            url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            # Protein (recommended) name
            protein_name = None
            desc = data.get("proteinDescription", {})
            rec = desc.get("recommendedName", {})
            if rec.get("fullName", {}).get("value"):
                protein_name = rec["fullName"]["value"]
            elif desc.get("submissionNames"):
                protein_name = desc["submissionNames"][0].get("fullName", {}).get("value")
            
            # Gene name
            gene_name = None
            genes = data.get("genes", [])
            if genes and genes[0].get("geneName", {}).get("value"):
                gene_name = genes[0]["geneName"]["value"]
            
            # Organism
            organism = data.get("organism", {}).get("scientificName")
            
            # Function (first function comment)
            function_text = None
            for comment in data.get("comments", []):
                if comment.get("commentType") == "FUNCTION":
                    texts = comment.get("texts", [])
                    if texts:
                        function_text = texts[0].get("value")
                        break
            
            return {
                "status": "success",
                "uniprot_id": uniprot_id,
                "protein_name": protein_name or "Unknown",
                "gene_name": gene_name or "Unknown",
                "organism": organism or "Unknown",
                "function": function_text or "No function description available",
                "source": "UniProt REST API"
            }
        
        except Exception as e:
            return {"error": str(e), "status": "failed",
                    "message": f"Could not retrieve info for {uniprot_id} from UniProt"}
    
    async def _validate_smiles(self, smiles: str) -> dict:
        """Validate SMILES string"""
        try:
            if not self.analyzer:
                return {"error": "MolecularSimilarityAnalyzer not available"}
            
            is_valid = self.analyzer.validate_smiles(smiles)
            
            return {
                "smiles": smiles,
                "is_valid": is_valid,
                "status": "success"
            }
        
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    async def _batch_extraction(self, uniprot_ids: list, output_format: str) -> dict:
        """Batch heteroatom extraction"""
        try:
            if not self.extractor:
                return {"error": "HeteroatomExtractor not available"}
            
            results = []
            for uniprot_id in uniprot_ids:
                pdb_ids = self.extractor.get_pdbs_for_uniprot(uniprot_id)
                results.append({
                    "uniprot_id": uniprot_id,
                    "pdb_count": len(pdb_ids),
                    "status": "success" if pdb_ids else "no_structures"
                })
            
            return {
                "status": "success",
                "total_proteins": len(uniprot_ids),
                "format": output_format,
                "results": results
            }
        
        except Exception as e:
            return {"error": str(e), "status": "failed"}


# Tool calling functions for MCP protocol
def get_available_tools() -> list:
    """Return list of available tools"""
    server = MCPServer()
    return server.define_tools()


async def call_tool(tool_name: str, tool_input: dict) -> Any:
    """Call a tool with given input"""
    server = MCPServer()
    return await server.execute_tool(tool_name, tool_input)


if __name__ == "__main__":
    # For testing the MCP server
    import asyncio
    
    async def test():
        server = MCPServer()
        tools = server.define_tools()
        print(f"Available tools: {len(tools)}")
        for tool in tools:
            print(f"  - {tool['name']}")
    
    asyncio.run(test())
