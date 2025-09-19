

# pages/home.py
from __future__ import annotations
import streamlit as st
from datetime import datetime


def render_home(cfg, oai):
    st.header("🏠 Home")
    st.caption("Quick overview & recent activity")

    # Recent uploads
    uploads = st.session_state.get("uploaded_paths", [])[-3:]
    if uploads:
        st.markdown("#### Recent RFP Uploads")
        for p in uploads:
            st.write(f"📄 {p}")
    else:
        st.info("No RFPs uploaded yet.")

    # Recent SharePoint sources
    sites = st.session_state.get("sp_chat_messages", [])
    if sites:
        st.markdown("#### Recent SharePoint Chats")
        for m in sites[-3:]:
            st.write(f"{m['role']}: {m['content'][:80]}…")

    # Last proposal generated
    last_meta = st.session_state.get("last_generation_meta")
    if last_meta:
        st.markdown("#### Last Proposal Generation")
        st.write(f"Template: {last_meta.get('template')}")
        st.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    else:
        st.info("No proposals generated yet.")
