"""Shared HTTP plumbing: one retrying session per thread, sensible timeouts and
an identifying User-Agent, as requested by the RCSB, PDBe and UniProt APIs."""

from __future__ import annotations

import threading

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_TIMEOUT = 30  # seconds
USER_AGENT = "TrackMyPDB (+https://trackmypdbsscai.streamlit.app)"

_local = threading.local()


def make_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    """Return a ``requests.Session`` that retries transient failures
    (connection errors, 429 and 5xx responses) with exponential backoff."""
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers["User-Agent"] = USER_AGENT
    return session


def get_session() -> requests.Session:
    """Per-thread session (``requests.Session`` is not guaranteed thread-safe)."""
    session = getattr(_local, "session", None)
    if session is None:
        session = make_session()
        _local.session = session
    return session
