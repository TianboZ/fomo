"""Fixed top header: logo + page navigation, shared by every page."""

from __future__ import annotations

import streamlit as st

HEADER_HEIGHT = 72  # px

_THEMES = {
    "dark": dict(bg="#1a1c24", border="rgba(250,250,250,0.12)", muted="rgba(250,250,250,0.65)",
                 text="#fafafa", hover="rgba(250,250,250,0.06)"),
    "light": dict(bg="#f0f2f6", border="rgba(49,51,63,0.12)", muted="rgba(49,51,63,0.65)",
                  text="#31333f", hover="rgba(49,51,63,0.06)"),
}
ACCENT = "#ff4b4b"


def _css(c: dict) -> str:
    h = HEADER_HEIGHT
    return f"""
<style>
.st-key-fomo_header {{
  position: fixed; top: 0; left: 0; right: 0; z-index: 999991;
  height: {h}px; padding: 0 2rem; background: {c['bg']}; overflow: hidden;
  border-bottom: 1px solid {c['border']};
}}
.stApp [data-testid="stHeader"] {{ top: {h}px; }}
[data-testid="stSidebar"] {{ top: {h}px; height: calc(100vh - {h}px); }}
.stMainBlockContainer {{ padding-top: calc(3rem + {h}px); }}
.st-key-fomo_header [data-testid="stPageLink-NavLink"] {{
  margin: 0; padding: 0.45rem 0.9rem; border-radius: 0.5rem;
  border: 1px solid transparent; background: transparent;
}}
.st-key-fomo_header [data-testid="stPageLink-NavLink"]:hover {{ background: {c['hover']}; }}
.st-key-fomo_header [data-testid="stPageLink-NavLink"] p {{ font-size: 1rem; color: {c['text']}; }}
.st-key-fomo_logo [data-testid="stPageLink-NavLink"] {{ padding: 0; }}
.st-key-fomo_logo [data-testid="stPageLink-NavLink"]:hover {{ background: transparent; }}
.st-key-fomo_logo [data-testid="stPageLink-NavLink"] p {{
  font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1;
}}
.st-key-fomo_logo [data-testid="stPageLink-NavLink"] p span {{ color: {ACCENT}; }}
.st-key-fomo_nav {{ border-left: 1px solid {c['border']}; padding-left: 1.5rem; }}
.st-key-fomo_nav_active [data-testid="stPageLink-NavLink"] {{
  background: rgba(255,75,75,0.15); border-color: rgba(255,75,75,0.35);
}}
.st-key-fomo_nav_active [data-testid="stPageLink-NavLink"] p {{ color: {ACCENT}; font-weight: 500; }}
.st-key-fomo_header_note p {{ color: {c['muted']}; font-size: 0.95rem; margin: 0; white-space: nowrap; }}
@media (max-width: 900px) {{
  .st-key-fomo_header {{ padding: 0 1rem; }}
  .st-key-fomo_header_note {{ display: none; }}
}}
</style>
"""


def render_header(home: st.Page, nav_pages: list[st.Page], current: st.Page) -> None:
    """Draw the header; `current` is the page returned by `st.navigation`."""
    colors = _THEMES["light" if st.context.theme.type == "light" else "dark"]
    st.html(_css(colors))
    with st.container(key="fomo_header", horizontal=True, horizontal_alignment="distribute",
                      vertical_alignment="center", gap="large", wrap=False):
        with st.container(horizontal=True, vertical_alignment="center", gap="large", width="content",
                          wrap=False):
            with st.container(key="fomo_logo", width="content"):
                st.page_link(home, label="你太:red[fomo]了")
            with st.container(key="fomo_nav", horizontal=True, vertical_alignment="center", width="content"):
                for page in nav_pages:
                    active = page.url_path == current.url_path
                    with st.container(key="fomo_nav_active" if active else f"fomo_nav_{page.url_path}",
                                      width="content"):
                        st.page_link(page)
        with st.container(key="fomo_header_note", width="content"):
            st.markdown("More functions coming soon")
