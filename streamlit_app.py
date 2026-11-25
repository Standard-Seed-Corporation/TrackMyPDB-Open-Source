"""
TrackMyPDB web app (Streamlit).

The app is a thin user interface over the ``trackmypdb`` library; each page
lives in ``ui/``. Run with::

    streamlit run streamlit_app.py

MIT License.
"""

import streamlit as st

try:  # optional: load ANTHROPIC_API_KEY etc. from a local .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="TrackMyPDB", page_icon="🧬", layout="wide")

try:
    import rdkit  # noqa: F401
except ImportError:
    st.error("RDKit is not installed. Install the requirements with "
             "`pip install -r requirements.txt` and restart the app.")
    st.stop()

from ui import common, database_search, disease, extraction, home, pipeline  # noqa: E402
from ui import similarity, style, visualizer  # noqa: E402


def assistant_page() -> None:
    try:
        from agent.streamlit_chat import render
    except ImportError as exc:
        st.error(f"The AI assistant needs the optional packages `anthropic` and `mcp` ({exc}).")
        return
    render()


PAGE_SPECS = [
    # key, render function, title, icon, url path
    ("home", home.render, "Home", ":material/home:", "home"),
    ("search", database_search.render, "Quick Search", ":material/bolt:", "quick-search"),
    ("pipeline", pipeline.render, "Target Search", ":material/my_location:", "target-search"),
    ("extraction", extraction.render, "Ligand Extraction", ":material/science:", "extraction"),
    ("similarity", similarity.render, "Similarity Ranking", ":material/compare_arrows:",
     "similarity"),
    ("visualizer", visualizer.render, "Molecule Viewer", ":material/hub:", "viewer"),
    ("disease", disease.render, "Disease Enrichment", ":material/health_and_safety:", "disease"),
    ("assistant", assistant_page, "AI Assistant", ":material/smart_toy:", "assistant"),
]

for key, fn, title, icon, path in PAGE_SPECS:
    common.PAGES[key] = st.Page(fn, title=title, icon=icon, url_path=path,
                                default=(key == "home"))

nav = st.navigation({
    "": [common.PAGES["home"]],
    "Search": [common.PAGES[k] for k in ("search", "pipeline")],
    "Step by step": [common.PAGES[k] for k in ("extraction", "similarity")],
    "Tools": [common.PAGES[k] for k in ("visualizer", "disease", "assistant")],
})
style.inject()
common.sidebar_branding()
nav.run()
