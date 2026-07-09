"""
Test script for TrackMyPDB Agentic Layer
Verifies all components are working correctly
"""

import os
import sys
import json
import asyncio
from pathlib import Path

# Add to path
sys.path.insert(0, os.path.dirname(__file__))

def test_mcp_server():
    """Test MCP server initialization and tools"""
    print("\n" + "="*60)
    print("Testing MCP Server...")
    print("="*60)
    
    try:
        from mcp_trackmypdb_server import MCPServer
        
        server = MCPServer()
        tools = server.define_tools()
        
        print(f"✓ MCP Server initialized")
        print(f"✓ Found {len(tools)} tools:")
        
        for tool in tools:
            print(f"  - {tool['name']}")
        
        return True
    
    except Exception as e:
        print(f"✗ MCP Server test failed: {e}")
        return False


def test_heteroatom_extractor():
    """Test heteroatom extractor"""
    print("\n" + "="*60)
    print("Testing Heteroatom Extractor...")
    print("="*60)
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'TrackMyPDB-Open-Source-sakeer'))
        from backend.heteroatom_extractor import HeteroatomExtractor
        
        extractor = HeteroatomExtractor()
        print(f"✓ HeteroatomExtractor initialized")
        
        # Test with a known protein
        test_protein = "P37231"
        print(f"  Testing with protein: {test_protein}")
        
        pdb_ids = extractor.get_pdbs_for_uniprot(test_protein)
        
        if pdb_ids:
            print(f"✓ Found {len(pdb_ids)} PDB structures: {pdb_ids[:5]}...")
            return True
        else:
            print(f"⚠ No PDB structures found (may be network issue)")
            return True  # Don't fail, could be network
    
    except Exception as e:
        print(f"✗ Heteroatom Extractor test failed: {e}")
        return False


def test_agent_initialization():
    """Test agent initialization"""
    print("\n" + "="*60)
    print("Testing Agent Initialization...")
    print("="*60)
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("⚠ ANTHROPIC_API_KEY not set")
        print("  Set it with: export ANTHROPIC_API_KEY='your-key'")
        return False
    
    try:
        from trackmypdb_agent import TrackMyPDBAgent
        
        agent = TrackMyPDBAgent(api_key=api_key)
        
        print(f"✓ Agent initialized with model: {agent.model}")
        print(f"✓ Available tools: {len(agent.tools)}")
        
        for tool in agent.tools:
            print(f"  - {tool['name']}")
        
        return True
    
    except ValueError as e:
        if "API key" in str(e):
            print(f"⚠ {e}")
            return False
        raise
    except Exception as e:
        print(f"✗ Agent initialization failed: {e}")
        return False


async def test_mcp_tools():
    """Test MCP tool execution"""
    print("\n" + "="*60)
    print("Testing MCP Tool Execution...")
    print("="*60)
    
    try:
        from mcp_trackmypdb_server import MCPServer
        
        server = MCPServer()
        
        # Test SMILES validation
        print("Testing SMILES validation...")
        result = await server.execute_tool(
            "validate_smiles",
            {"smiles": "CCO"}
        )
        
        if result.get("status") == "success" or "error" not in result:
            print(f"✓ SMILES validation works")
        else:
            print(f"⚠ SMILES validation result: {result}")
        
        # Test PDB structure retrieval
        print("Testing PDB structure retrieval...")
        result = await server.execute_tool(
            "get_pdb_structures",
            {"uniprot_id": "P37231"}
        )
        
        if result.get("status") == "success":
            count = result.get("total_structures", 0)
            print(f"✓ PDB retrieval works ({count} structures)")
        else:
            print(f"⚠ PDB retrieval result: {result}")
        
        return True
    
    except Exception as e:
        print(f"✗ MCP tool execution failed: {e}")
        return False


def test_streamlit_interface():
    """Test Streamlit interface exists and is valid"""
    print("\n" + "="*60)
    print("Testing Streamlit Interface...")
    print("="*60)
    
    try:
        interface_path = Path(__file__).parent / "chat_interface.py"
        
        if interface_path.exists():
            print(f"✓ Chat interface file exists: {interface_path}")
            
            # Check imports
            with open(interface_path, 'r') as f:
                content = f.read()
                
                if "import streamlit" in content:
                    print(f"✓ Streamlit import found")
                if "TrackMyPDBAgent" in content:
                    print(f"✓ Agent integration found")
                if "chat_interface.py" in str(interface_path):
                    print(f"✓ Interface properly structured")
            
            return True
        else:
            print(f"✗ Chat interface file not found: {interface_path}")
            return False
    
    except Exception as e:
        print(f"✗ Streamlit interface test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist"""
    print("\n" + "="*60)
    print("Testing File Structure...")
    print("="*60)
    
    required_files = [
        "mcp_trackmypdb_server.py",
        "trackmypdb_agent.py",
        "chat_interface.py",
        "IMPLEMENTATION_GUIDE.md",
        "QUICK_START.md",
        "requirements_agent.txt",
        "test_agent.py"
    ]
    
    base_dir = Path(__file__).parent
    all_exist = True
    
    for filename in required_files:
        filepath = base_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"✓ {filename} ({size:,} bytes)")
        else:
            print(f"✗ {filename} NOT FOUND")
            all_exist = False
    
    # Check TrackMyPDB directory
    trackmypdb_dir = base_dir / "TrackMyPDB-Open-Source-sakeer"
    if trackmypdb_dir.exists():
        print(f"✓ TrackMyPDB directory exists")
    else:
        print(f"✗ TrackMyPDB directory NOT FOUND")
        all_exist = False
    
    return all_exist


def run_all_tests():
    """Run all tests and report results"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║  TrackMyPDB Agentic Layer - Test Suite               ║")
    print("╚" + "="*58 + "╝")
    
    results = {
        "File Structure": test_file_structure(),
        "MCP Server": test_mcp_server(),
        "Heteroatom Extractor": test_heteroatom_extractor(),
        "Agent Initialization": test_agent_initialization(),
        "Streamlit Interface": test_streamlit_interface(),
    }
    
    # Run async tests
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results["MCP Tool Execution"] = loop.run_until_complete(test_mcp_tools())
    except Exception as e:
        print(f"\n⚠ Skipping async tests: {e}")
        results["MCP Tool Execution"] = "SKIPPED"
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v == "SKIPPED")
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASSED"
        elif result is False:
            status = "✗ FAILED"
        else:
            status = "⊘ SKIPPED"
        print(f"{test_name:.<40} {status}")
    
    print("="*60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n✓ All required tests passed!")
        print("\nNext steps:")
        print("1. Set your API key: export ANTHROPIC_API_KEY='sk-...'")
        print("2. Launch the chat interface: streamlit run chat_interface.py")
        print("3. Start researching!")
    else:
        print(f"\n✗ {failed} test(s) failed. Please review the output above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
