# modules/ui_layout.py
from __future__ import annotations
import streamlit as st

def setup_branding(
    title: str = "Proposal Studio — RFP & SharePoint",
    tagline: str = "Ingest • Understand • Compose — proposals at speed",
):
    st.set_page_config(page_title=title, page_icon="📄", layout="wide")

    st.markdown("""
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 4rem;}
    .app-hero {text-align:center; margin-bottom: .5rem; font-size: 2.0rem; font-weight: 700;}
    .app-sub {text-align:center; color: rgba(128,128,128,.9); margin-bottom: 1.2rem;}
    .stTabs [data-baseweb="tab"] {font-weight:600;}
    .status-card { border:1px solid rgba(0,0,0,.08); border-radius:12px; padding:14px 16px; background:rgba(0,0,0,.02); }
    .status-k {font-size:.82rem; color: rgba(0,0,0,.55);}
    .status-v {font-weight:600;}
    .qa {display:flex; gap:12px; flex-wrap:wrap; margin:.25rem 0 1rem 0}
    .qa .qa-btn {padding:10px 14px; border-radius:10px; border:1px solid rgba(0,0,0,.08);}
    .qa .qa-btn:hover {background: rgba(0,0,0,.03);}
    .sticky { position: sticky; top: 0; z-index: 9; background: var(--background-color); padding: .4rem 0 .6rem 0; }
    .badge { display:inline-block; padding:.15rem .5rem; border-radius:999px; font-size:.75rem; background:rgba(3,158,237,.12); color:#039eed; border:1px solid rgba(3,158,237,.25); }
    section[data-testid="stChatMessage"] { margin-bottom: .35rem; }
    .smallcap { font-size:.8rem; opacity:.8; }
    </style>
    """, unsafe_allow_html=True)

    # sidebar jump + anchor
    st.sidebar.markdown("[🏠 Home](#home)", unsafe_allow_html=True)
    st.markdown('<a id="home"></a>', unsafe_allow_html=True)

    # hero
    st.markdown(f'<div class="app-hero">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-sub">{tagline}</div>', unsafe_allow_html=True)
