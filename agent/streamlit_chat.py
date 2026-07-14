"""
In-app Claude chat tab for TrackMyPDB.

render() is called from streamlit_app.py's show_ai_assistant_page(). It gives
users a chat box that talks to Claude, which in turn drives the TrackMyPDB MCP
server tools.

MIT License - Open Source Project.
"""

from __future__ import annotations

import os

import streamlit as st

from agent.mcp_agent import run_agent, DEFAULT_MODEL


def _get_api_key():
    try:
        secret = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        secret = None
    return secret or os.getenv("ANTHROPIC_API_KEY") or st.session_state.get("anthropic_api_key")


def render():
    st.header("TrackMyPDB AI Assistant")
    st.caption("Chat with your data. Powered by Claude + a Model Context Protocol (MCP) server.")

    api_key = _get_api_key()
    with st.sidebar:
        st.markdown("### AI Assistant settings")
        if not api_key:
            entered = st.text_input(
                "Anthropic API key", type="password",
                help="Create one at console.anthropic.com. Kept only in this browser session.")
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
            "- For UniProt Q9UNQ0, which ligands are most similar to aspirin "
            "(CC(=O)OC1=CC=CC=C1C(=O)O)?\n"
            "- How similar are caffeine and theophylline?")

    for msg in st.session_state["chat_display"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

    prompt = st.chat_input("Ask about proteins, ligands, or molecular similarity...")
    if not prompt:
        return

    st.session_state["chat_display"].append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not api_key:
        with st.chat_message("assistant"):
            st.warning("Enter your Anthropic API key in the sidebar to start chatting.")
        st.session_state["chat_display"].append(
            {"role": "assistant", "text": "_(Waiting for an Anthropic API key.)_"})
        return

    st.session_state["chat_messages"].append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        status = st.status("Thinking and calling TrackMyPDB tools...", expanded=True)

        def on_event(ev):
            if ev["type"] == "tool_call":
                status.write("Calling `%s` with %s" % (ev["name"], ev["input"]))
            elif ev["type"] == "tool_result":
                status.write("`%s` returned %d chars" % (ev["name"], len(str(ev["content"]))))

        try:
            final_text, updated = run_agent(
                st.session_state["chat_messages"], api_key=api_key,
                model=model, on_event=on_event)
            st.session_state["chat_messages"] = updated
            status.update(label="Done", state="complete", expanded=False)
            st.markdown(final_text)
            st.session_state["chat_display"].append({"role": "assistant", "text": final_text})
        except Exception as exc:  # noqa: BLE001
            status.update(label="Error", state="error")
            st.error("Agent error: %s" % exc)
            st.session_state["chat_display"].append(
                {"role": "assistant", "text": "Sorry, something went wrong: %s" % exc})
