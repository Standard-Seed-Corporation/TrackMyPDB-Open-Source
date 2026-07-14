"""
Streamlit shim for TrackMyPDB.

The backend modules (backend/heteroatom_extractor.py and
backend/similarity_analyzer.py) call `import streamlit as st` and render UI with
st.info / st.progress / st.columns etc.

The MCP server runs OUTSIDE any Streamlit runtime (a plain Python process
speaking JSON-RPC over stdio). If the backend imported the real streamlit there,
every st.* call would print a "missing ScriptRunContext" warning and could even
write to stdout and corrupt the MCP protocol stream.

This module is a drop-in replacement that turns every st.* call into a silent
no-op. core.py installs it into sys.modules['streamlit'] BEFORE importing the
backend, so the backend runs unchanged with zero UI side effects.

MIT License - Open Source Project.
"""

from __future__ import annotations


class _Noop:
    """Swallows any attribute access, call, or context-manager use."""

    def __call__(self, *args, **kwargs):
        return _Noop()

    def __getattr__(self, _name):
        return _Noop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(())

    def progress(self, *args, **kwargs):
        return _Noop()

    def empty(self, *args, **kwargs):
        return _Noop()

    def text(self, *args, **kwargs):
        return _Noop()


# --- functions where the SHAPE of the return value matters ---------------------

def columns(spec, **kwargs):
    """Mirror st.columns so `a, b = st.columns(2)` still unpacks correctly."""
    n = spec if isinstance(spec, int) else len(spec)
    return [_Noop() for _ in range(n)]


def tabs(labels, **kwargs):
    return [_Noop() for _ in labels]


def progress(*args, **kwargs):
    return _Noop()


def empty(*args, **kwargs):
    return _Noop()


def spinner(*args, **kwargs):
    return _Noop()


def expander(*args, **kwargs):
    return _Noop()


def container(*args, **kwargs):
    return _Noop()


# namespaces / objects the backend touches by attribute
sidebar = _Noop()
column_config = _Noop()
session_state = {}


def __getattr__(name):
    """PEP 562: anything not defined above becomes a silent no-op callable."""
    return _Noop()
