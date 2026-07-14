"""
Claude agent that connects to the TrackMyPDB MCP server.

Flow for each user message:
  1. Launch mcp_server.server over stdio (official MCP Python client).
  2. Discover the server's tools and hand them to Claude.
  3. Run the tool-use loop: Claude thinks -> calls a tool -> reads the result ->
     ... -> final answer.

This is what makes the app "agentic AND MCP-native": the same MCP server that
powers Claude Desktop also powers the in-app chat.

MIT License - Open Source Project.
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import AsyncExitStack

from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

SYSTEM_PROMPT = (
    "You are TrackMyPDB Assistant, a friendly bioinformatics helper for the "
    "TrackMyPDB app. You can list PDB structures for a UniProt ID, extract "
    "drug-like heteroatoms (ligands) and their SMILES from PDB structures, and "
    "rank molecules by Tanimoto similarity using Morgan fingerprints -- all via "
    "the provided tools. When the user gives UniProt IDs together with a target "
    "molecule, use run_pipeline. Keep max_pdbs_per_uniprot small (<=5) unless "
    "asked, because extraction calls slow public APIs (RCSB, PDBe, PubChem). "
    "Explain the chemistry in plain language and always show the top hits."
)


def _server_params() -> StdioServerParameters:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        env=env,
    )


def _tool_result_text(result) -> str:
    parts = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts) if parts else str(result)


async def _run(messages, api_key, model, on_event):
    client = Anthropic(api_key=api_key)
    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(_server_params()))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        tools = (await session.list_tools()).tools
        anthropic_tools = [{
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.inputSchema,
        } for t in tools]

        while True:
            resp = client.messages.create(
                model=model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=anthropic_tools,
                messages=messages,
            )

            assistant_content = []
            tool_uses = []
            for block in resp.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use", "id": block.id,
                        "name": block.name, "input": block.input,
                    })
                    tool_uses.append(block)
            messages.append({"role": "assistant", "content": assistant_content})

            if resp.stop_reason != "tool_use":
                final = "".join(b.text for b in resp.content if b.type == "text")
                return final, messages

            tool_results = []
            for tu in tool_uses:
                if on_event:
                    on_event({"type": "tool_call", "name": tu.name, "input": tu.input})
                try:
                    result = await session.call_tool(tu.name, tu.input or {})
                    text = _tool_result_text(result)
                    is_error = bool(getattr(result, "isError", False))
                except Exception as exc:  # noqa: BLE001
                    text, is_error = f"Tool error: {exc}", True
                if on_event:
                    on_event({"type": "tool_result", "name": tu.name, "content": text})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": text,
                    "is_error": is_error,
                })
            messages.append({"role": "user", "content": tool_results})


def run_agent(messages, api_key=None, model=None, on_event=None):
    """
    Synchronous entry point used by the Streamlit chat.

    messages : Anthropic-format message list (mutated in place and returned).
    Returns  : (final_answer_text, updated_messages)
    """
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")
    model = model or DEFAULT_MODEL
    return asyncio.run(_run(messages, api_key, model, on_event))
