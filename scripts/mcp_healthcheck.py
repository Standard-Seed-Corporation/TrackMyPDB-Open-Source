"""
MCP connection health check for TrackMyPDB.

Connects to mcp_server.server exactly the way the in-app agent does, lists the
tools it exposes, and calls one offline tool (compare_smiles) to prove the whole
client <-> server <-> backend chain works -- no Anthropic key and no internet
required.

Run:
    py -3.11 scripts/mcp_healthcheck.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _server_params() -> StdioServerParameters:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        env=env,
    )


async def main() -> int:
    print("Launching MCP server and connecting...")
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = (await session.list_tools()).tools
            print(f"\nConnected. Server exposes {len(tools)} tools:")
            for tool in tools:
                print(f"   - {tool.name}")

            print("\nCalling compare_smiles('CCO', 'CCO') as a smoke test...")
            result = await session.call_tool("compare_smiles",
                                             {"smiles_a": "CCO", "smiles_b": "CCO"})
            text = "".join(getattr(b, "text", "") for b in (result.content or []))
            print("   ->", text)

            ok = '"tanimoto_similarity": 1.0' in text or "1.0" in text
            print("\nHEALTH CHECK:", "PASS " * 0 + ("PASS" if ok else "CHECK OUTPUT"))
            return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
