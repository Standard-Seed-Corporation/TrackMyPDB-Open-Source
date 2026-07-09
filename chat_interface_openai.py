"""
TrackMyPDB Chat Interface - OpenAI Version
Web-based UI for interacting with the intelligent agent powered by OpenAI
Built with Streamlit for ease of deployment and use
"""

import streamlit as st
import json
import os
from datetime import datetime
from pathlib import Path
import sys

# Add paths
sys.path.append(os.path.dirname(__file__))

from trackmypdb_agent_openai import TrackMyPDBAgent, AgentSession, create_protein_analysis_workflow

# Configure Streamlit
st.set_page_config(
    page_title="TrackMyPDB Agent Chat - OpenAI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1a1a1a;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1em;
        margin-bottom: 20px;
    }
    .message-user {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
        padding: 12px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .message-assistant {
        background-color: #f5f5f5;
        border-left: 4px solid #4CAF50;
        padding: 12px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .tool-call {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 10px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "agent" not in st.session_state:
    try:
        st.session_state.agent = TrackMyPDBAgent()
        st.session_state.session_initialized = True
    except ValueError as e:
        st.error(f"Error initializing agent: {e}")
        st.error("Please set OPENAI_API_KEY environment variable")
        st.session_state.session_initialized = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "research_context" not in st.session_state:
    st.session_state.research_context = {
        "proteins": [],
        "ligands": [],
        "analyses_performed": []
    }


def update_research_context(context_update: dict):
    """Update research context"""
    st.session_state.research_context.update(context_update)


def add_to_context_list(key: str, value):
    """Add item to context list"""
    if key not in st.session_state.research_context:
        st.session_state.research_context[key] = []
    if value not in st.session_state.research_context[key]:
        st.session_state.research_context[key].append(value)


def format_message(role: str, content: str, timestamp: str = None):
    """Format a message for display"""
    if role == "user":
        st.markdown(f'<div class="message-user"><b>You:</b> {content}</div>', unsafe_allow_html=True)
    elif role == "assistant":
        st.markdown(f'<div class="message-assistant"><b>Agent:</b> {content}</div>', unsafe_allow_html=True)


def save_chat_session(filepath: str = None):
    """Save the chat session"""
    if filepath is None:
        filepath = f"chat_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    session_data = {
        "timestamp": datetime.now().isoformat(),
        "messages": st.session_state.agent.get_history(),
        "research_context": st.session_state.research_context
    }
    
    with open(filepath, 'w') as f:
        json.dump(session_data, f, indent=2)
    
    return filepath


def load_chat_session(filepath: str):
    """Load a previous chat session"""
    with open(filepath, 'r') as f:
        session_data = json.load(f)
    
    st.session_state.agent.load_conversation(filepath)
    st.session_state.research_context = session_data.get("research_context", {})


# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="main-title">🧬 TrackMyPDB Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Intelligent Bioinformatics Research Assistant - OpenAI Version</div>', unsafe_allow_html=True)

st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    if not os.getenv("OPENAI_API_KEY"):
        api_key = st.text_input("OpenAI API Key", type="password", key="api_key_input")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.session_state.agent = TrackMyPDBAgent(api_key=api_key)
    else:
        st.success("✓ API Key configured")
    
    st.divider()
    
    # Research Context
    st.subheader("📋 Research Context")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📊 View Context", use_container_width=True):
            st.json(st.session_state.research_context)
    
    with col2:
        if st.button("🗑️ Clear Context", use_container_width=True):
            st.session_state.research_context = {
                "proteins": [],
                "ligands": [],
                "analyses_performed": []
            }
            st.rerun()
    
    st.divider()
    
    # Session Management
    st.subheader("💾 Session Management")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("💾 Save Chat", use_container_width=True):
            filepath = save_chat_session()
            st.success(f"Saved to {filepath}")
    
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.agent.clear_history()
            st.session_state.messages = []
            st.rerun()
    
    st.divider()
    
    # Quick Templates
    st.subheader("📝 Quick Templates")
    
    template_option = st.selectbox(
        "Choose a template or start custom:",
        [
            "Custom",
            "Protein Analysis",
            "Ligand Similarity",
            "Structure Extraction",
            "Batch Analysis"
        ],
        key="template_selector"
    )
    
    template_inputs = {}
    
    if template_option == "Protein Analysis":
        proteins = st.text_area("UniProt IDs (comma-separated):", height=100)
        if proteins:
            template_inputs["uniprot_ids"] = [p.strip() for p in proteins.split(",")]
    
    elif template_option == "Ligand Similarity":
        smiles = st.text_input("Target SMILES:")
        proteins = st.text_area("Protein IDs (comma-separated):", height=80)
        if smiles and proteins:
            template_inputs["target_smiles"] = smiles
            template_inputs["uniprot_ids"] = [p.strip() for p in proteins.split(",")]
    
    elif template_option == "Structure Extraction":
        protein = st.text_input("UniProt ID:")
        if protein:
            template_inputs["uniprot_id"] = protein
    
    elif template_option == "Batch Analysis":
        proteins = st.text_area("UniProt IDs (one per line):", height=120)
        if proteins:
            template_inputs["uniprot_ids"] = [p.strip() for p in proteins.split("\n") if p.strip()]


# Main chat area
st.header("💬 Conversation")

# Display chat history
if not st.session_state.session_initialized:
    st.warning("Agent not initialized. Please check API key configuration.")
else:
    # Display previous messages
    for msg in st.session_state.agent.get_history():
        format_message(msg["role"], msg["content"], msg.get("timestamp"))
    
    st.divider()
    
    # User input
    col1, col2 = st.columns([0.95, 0.05])
    
    with col1:
        user_input = st.text_area(
            "Your message:",
            placeholder="Ask me about protein structures, ligand discovery, molecular similarity...",
            height=80,
            key="user_input"
        )
    
    with col2:
        send_button = st.button("📤", help="Send message", use_container_width=True)
    
    # Send message
    if send_button and user_input:
        # Add user message
        format_message("user", user_input)
        
        # Show loading indicator
        with st.spinner("Agent is thinking..."):
            try:
                # Get response from agent
                response = st.session_state.agent.chat(user_input)
                
                # Display response
                format_message("assistant", response)
                
                # Update context
                if any(keyword in user_input.lower() for keyword in ["protein", "uniprot", "pdb"]):
                    add_to_context_list("analyses_performed", f"Protein analysis - {datetime.now().isoformat()}")
                
                if any(keyword in user_input.lower() for keyword in ["similarity", "similar", "smiles"]):
                    add_to_context_list("analyses_performed", f"Similarity analysis - {datetime.now().isoformat()}")
                
                st.success("Message processed successfully!")
                st.rerun()
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error("Please ensure your API key is valid and you have internet connection.")


# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.9em;'>
    <p>TrackMyPDB Intelligent Agent - OpenAI Version • Powered by OpenAI GPT-4 & MCP Tools</p>
    <p>© 2025 Standard Seed Corporation • <a href='https://github.com/Standard-Seed-Corporation/TrackMyPDB-Open-Source'>GitHub</a></p>
</div>
""", unsafe_allow_html=True)
