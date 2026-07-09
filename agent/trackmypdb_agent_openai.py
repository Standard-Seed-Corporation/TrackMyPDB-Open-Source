"""
TrackMyPDB Intelligent Agent - OpenAI Version
Uses OpenAI's GPT-4 instead of Claude API
"""

import json
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from openai import OpenAI
from mcp_trackmypdb_server import MCPServer


@dataclass
class Message:
    """Represents a message in the conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class TrackMyPDBAgent:
    """
    Intelligent agent for TrackMyPDB research using OpenAI
    Uses GPT-4 with tool calling for specialized functions
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize the agent
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: OpenAI model to use (gpt-4, gpt-4-turbo, gpt-3.5-turbo)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.mcp_server = MCPServer()
        self.tools = self._convert_mcp_to_openai_tools()
        self.conversation_history: List[Message] = []
    
    def _convert_mcp_to_openai_tools(self) -> list:
        """Convert MCP tool definitions to OpenAI format"""
        mcp_tools = self.mcp_server.define_tools()
        openai_tools = []
        
        for tool in mcp_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["inputSchema"]
                }
            }
            openai_tools.append(openai_tool)
        
        return openai_tools
    
    def _create_system_prompt(self) -> str:
        """Create a system prompt for the agent"""
        return """You are an expert bioinformatics research assistant powered by TrackMyPDB and OpenAI.
        
Your capabilities:
1. Extract heteroatoms from PDB protein structures
2. Perform molecular similarity analysis using Morgan fingerprints
3. Access PDB structure databases
4. Validate molecular SMILES representations
5. Perform batch bioinformatics operations

You have access to specialized tools for protein structure analysis and molecular chemistry.
When users ask about protein analysis, ligand discovery, or molecular similarity, use the available tools.

Be conversational, explain your findings clearly, and provide actionable insights.
Always explain the scientific significance of results.

For complex requests:
1. Break down the analysis into steps
2. Use appropriate tools for each step
3. Synthesize results into coherent findings
4. Suggest next steps or related analyses"""
    
    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response from the agent
        
        Args:
            user_message: The user's input message
            
        Returns:
            The assistant's response
        """
        # Add user message to history
        self.conversation_history.append(Message(role="user", content=user_message))
        
        # Prepare conversation for API
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in self.conversation_history
        ]
        
        # Get response from OpenAI with tools
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )
        
        # Process response and handle tool calls
        assistant_message = ""
        tool_calls = []
        
        # Extract content
        for choice in response.choices:
            if choice.message.content:
                assistant_message += choice.message.content
            
            # Check for tool calls
            if choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "input": json.loads(tool_call.function.arguments)
                    })
        
        # Process tool calls if any
        if tool_calls:
            # Add assistant message with tool calls
            self.conversation_history.append(
                Message(role="assistant", content=assistant_message)
            )
            
            # Execute tools and get results
            tool_results = []
            for tool_call in tool_calls:
                try:
                    tool_result = self._execute_tool(
                        tool_call["name"],
                        tool_call["input"]
                    )
                    tool_results.append({
                        "tool_use_id": tool_call["id"],
                        "content": json.dumps(tool_result)
                    })
                except Exception as e:
                    tool_results.append({
                        "tool_use_id": tool_call["id"],
                        "content": json.dumps({"error": str(e)})
                    })
            
            # Prepare tool results for follow-up
            messages.append({"role": "assistant", "content": response.choices[0].message.content})
            
            # Add tool results as user message
            tool_result_content = "Tool results:\n"
            for result in tool_results:
                tool_result_content += f"\n{result['content']}"
            
            messages.append({"role": "user", "content": tool_result_content})
            
            # Get final response after tool calls
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            assistant_message = final_response.choices[0].message.content
        
        # Add assistant response to history
        self.conversation_history.append(
            Message(role="assistant", content=assistant_message)
        )
        
        return assistant_message
    
    def _execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        """Execute a tool synchronously"""
        import asyncio
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.mcp_server.execute_tool(tool_name, tool_input)
            )
            return result
        finally:
            loop.close()
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history as list of dicts"""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            }
            for msg in self.conversation_history
        ]
    
    def save_conversation(self, filepath: str) -> None:
        """Save conversation to a JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.get_history(), f, indent=2)
    
    def load_conversation(self, filepath: str) -> None:
        """Load conversation from a JSON file"""
        with open(filepath, 'r') as f:
            history_data = json.load(f)
        
        self.conversation_history = [
            Message(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg.get("timestamp")
            )
            for msg in history_data
        ]


class AgentSession:
    """Manages a user session with the agent"""
    
    def __init__(self, session_id: str, api_key: Optional[str] = None):
        """
        Initialize a session
        
        Args:
            session_id: Unique identifier for the session
            api_key: OpenAI API key
        """
        self.session_id = session_id
        self.agent = TrackMyPDBAgent(api_key=api_key)
        self.created_at = datetime.now()
        self.research_context = {}
    
    def set_research_context(self, context: Dict[str, Any]) -> None:
        """Set research context for the session"""
        self.research_context.update(context)
    
    def chat(self, user_message: str) -> str:
        """Chat with the agent"""
        return self.agent.chat(user_message)
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "message_count": len(self.agent.conversation_history),
            "research_context": self.research_context
        }


# Utility functions for common workflows

def create_protein_analysis_workflow(uniprot_ids: List[str]) -> str:
    """Create a workflow prompt for protein analysis"""
    ids_str = ", ".join(uniprot_ids)
    return f"""Please analyze these UniProt proteins: {ids_str}

1. Extract heteroatoms from their PDB structures
2. Identify any interesting ligands or cofactors
3. Summarize the findings"""


def create_ligand_discovery_workflow(target_smiles: str, uniprot_ids: List[str]) -> str:
    """Create a workflow prompt for ligand discovery"""
    return f"""I'm looking for compounds similar to: {target_smiles}

Please:
1. Find heteroatoms from these proteins: {', '.join(uniprot_ids)}
2. Compare their structures to my target molecule
3. Identify the most similar compounds
4. Explain the structural similarities"""


def create_structure_analysis_workflow(uniprot_id: str) -> str:
    """Create a workflow prompt for structure analysis"""
    return f"""Please provide a comprehensive analysis of protein {uniprot_id}:

1. Find all available PDB structures
2. Extract heteroatoms from these structures
3. Analyze the ligands and cofactors
4. Suggest similar compounds"""


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Create agent
    agent = TrackMyPDBAgent()
    
    print("TrackMyPDB Intelligent Agent - OpenAI Version")
    print("=" * 50)
    print("Type 'quit' to exit, 'history' to see chat history")
    print()
    
    # Interactive chat loop
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == "quit":
            break
        elif user_input.lower() == "history":
            history = agent.get_history()
            for msg in history:
                print(f"\n[{msg['timestamp']}] {msg['role'].upper()}:")
                print(msg['content'])
            continue
        elif not user_input:
            continue
        
        print("\nAssistant: ", end="", flush=True)
        response = agent.chat(user_input)
        print(response)
        print()
