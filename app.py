"""Streamlit entrypoint and router: `streamlit run app.py`."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from views.header import render_header

st.set_page_config(page_title="你太fomo了", layout="wide")

lrs = st.Page(ROOT / "views" / "lrs.py", title="Leveraged Rotation Strategy", url_path="lrs")


def home_redirect() -> None:
    """No landing page yet: "/" forwards to the only tool."""
    st.switch_page(lrs)


home = st.Page(home_redirect, title="Home", default=True)
NAV_PAGES = [lrs]

current = st.navigation([home, *NAV_PAGES], position="hidden")
render_header(home, NAV_PAGES, current)
current.run()
