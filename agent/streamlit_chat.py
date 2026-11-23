"""
In-app AI assistant page: a chat with Claude, which calls the TrackMyPDB MCP
server tools.

API-key policy: by default every visitor supplies their own Anthropic key
(kept only in their browser session). A key stored in the deployment's
secrets/environment is used only when ``TRACKMYPDB_ALLOW_SHARED_KEY=1`` is
also set, so a public deployment does not silently spend the owner's credits.

MIT License.
"""

from __future__ import annotations

import os

import streamlit as st

from agent.mcp_agent import DEFAULT_MODEL, run_agent


def _secret(name: str):
    try:
        return st.secrets.get(name)
    except Exception:  # noqa: BLE001 - no secrets file
        return None


def _shared_key():
    allow = str(_secret("TRACKMYPDB_ALLOW_SHARED_KEY") or os.getenv("TRACKMYPDB_ALLOW_SHARED_KEY", ""))
    if allow.lower() in {"1", "true", "yes"}:
        return _secret("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    return None


def render() -> None:
    from ui.style import page_header
    page_header("AI Assistant", "Ask in plain language. Claude runs the tools for you.", icon="🤖")

    api_key = _shared_key() or st.session_state.get("anthropic_api_key")
    with st.sidebar:
        st.markdown("### Assistant settings")
        if not _shared_key():
            entered = st.text_input("Anthropic API key", type="password",
                                    value=st.session_state.get("anthropic_api_key", ""),
                                    help="Kept only in this browser session; never stored.")
            if entered:
                st.session_state["anthropic_api_key"] = entered
                api_key = entered
        model = st.text_input("Model", value=DEFAULT_MODEL,
                              help="Any model your Anthropic account can access.")
        if st.button("Clear chat"):
            st.session_state["chat_messages"] = []
            st.session_state["chat_display"] = []
            st.rerun()

    st.session_state.setdefault("chat_messages", [])   # Anthropic-format history
    st.session_state.setdefault("chat_display", [])    # [{role, text}] for rendering

    with st.expander("Example prompts", expanded=not st.session_state["chat_display"]):
        st.markdown(
            "- List the PDB structures for UniProt P37231.\n"
            "- Extract the ligands from UniProt P06276 (max 3 structures).\n"
            "- Which PDB ligands are most similar to aspirin, CC(=O)Oc1ccccc1C(=O)O?\n"
            "- For UniProt P37231, which ligands are most similar to rosiglitazone?\n"
            "- How similar are caffeine and theophylline?")

    for msg in st.session_state["chat_display"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

    prompt = st.chat_input("Ask about proteins, ligands or molecular similarity...")
    if not prompt:
        return

    st.session_state["chat_display"].append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not api_key:
        with st.chat_message("assistant"):
            st.warning("Enter your Anthropic API key in the sidebar to start chatting.")
        st.session_state["chat_display"].pop()
        return

    history = st.session_state["chat_messages"]
    checkpoint = len(history)
    history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        status = st.status("Thinking and calling TrackMyPDB tools...", expanded=True)

        def on_event(ev):
            if ev["type"] == "tool_call":
                status.write(f"Calling `{ev['name']}` with `{ev['input']}`")
            elif ev["type"] == "tool_result":
                status.write(f"`{ev['name']}` returned {len(str(ev['content'])):,} characters")

        try:
            final_text, updated = run_agent(history, api_key=api_key, model=model,
                                            on_event=on_event)
            st.session_state["chat_messages"] = updated
            status.update(label="Done", state="complete", expanded=False)
            st.markdown(final_text)
            st.session_state["chat_display"].append({"role": "assistant", "text": final_text})
        except Exception as exc:  # noqa: BLE001
            # Roll back this turn so the conversation history stays valid.
            del history[checkpoint:]
            status.update(label="Error", state="error")
            st.error(f"Assistant error: {exc}")
            st.session_state["chat_display"].append(
                {"role": "assistant", "text": f"_Something went wrong: {exc}_"})
