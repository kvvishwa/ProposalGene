# -----------------------------------------------------------------------------
# main.py (entry)
# -----------------------------------------------------------------------------
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
import streamlit as st
from openai import OpenAI

# Pages
from pages.home import render_home
from pages.understanding import render_understanding
from pages.chats import render_sharepoint_chat
from pages.generation import render_generation
from pages.settings_page import render_settings

# Modules
from modules.app_helpers import ensure_dirs

import json, hashlib
import streamlit as st

from modules import llm_processing
from modules.app_helpers import st_rerun_compat

from dotenv import load_dotenv
load_dotenv()   # <---- add this at the top of main.py, before Config()


@dataclass
class Config:
    # OpenAI
    ANALYSIS_MODEL: str = os.getenv("ANALYSIS_MODEL", "gpt-4o")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2200"))

    # SharePoint creds
    SP_CLIENT_ID: str = os.getenv("SHAREPOINT_CLIENT_ID", "")
    SP_CLIENT_SECRET: str = os.getenv("SHAREPOINT_CLIENT_SECRET", "")
    SP_TENANT_ID: str = os.getenv("SHAREPOINT_TENANT_ID", "")
    SP_SITE_URL: str = os.getenv("SHAREPOINT_SITE_URL", "")
    SP_AUTHORITY: str = os.getenv("SHAREPOINT_AUTHORITY","")

    # Vectorization
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))


# --- App constants / folders ---
BASE_DIR = Path("data")
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static_sources"
RFP_FACTS_DIR = BASE_DIR / "rfp_facts"
STATIC_MAP_FILE = BASE_DIR / "static_map.json"
SUPPORTED_UPLOAD_TYPES = ["pdf","doc","docx","ppt","pptx","xlsx","xls","txt"]

# Keep in sync with Generation page expectations
STATIC_SECTIONS = [
    "Cover Letter","Company Overview","Why BCT","Case Studies","Tools & Accelerators","Methodology Overview",
    "Project Governance","Quality Management","Risk Management","Security & Compliance","Change Management",
    "Transition & Knowledge Transfer","Assumptions & Exclusions","Commercials & Terms","Contact Details"
]

def _hash_facts(facts: dict) -> str:
    try:
        s = json.dumps(facts or {}, sort_keys=True, ensure_ascii=False)
    except Exception:
        s = str(facts or "")
    return hashlib.sha1(s.encode("utf-8")).hexdigest()



def _init_session_defaults():
    ss = st.session_state
    ss.setdefault("uploaded_paths", [])
    ss.setdefault("temp_files", [])
    ss.setdefault("vectorized", False)
    ss.setdefault("up_store", None)
    ss.setdefault("rfp_facts", None)
    ss.setdefault("rfp_raw", "")
    ss.setdefault("rfp_chat_messages", [])
    ss.setdefault("rfp_chat_last_evidence", [])
    ss.setdefault("sp_chat_messages", [])
    ss.setdefault("sp_chat_last_evidence", [])
    ss.setdefault("gen_tpl", "— choose a template —")
    ss.setdefault("gen_static_sel", [])
    ss.setdefault("gen_dyn_sel", [])
    ss.setdefault("gen_final_order_widget", [])
    ss.setdefault("gen_use_anchors", True)
    ss.setdefault("gen_include_sources", True)
    ss.setdefault("gen_top_k", 6)
    ss.setdefault("gen_rec_style", "paragraphs")
    ss.setdefault("gen_per_section_k", 25)
    ss.setdefault("gen_page_breaks", True)
    ss.setdefault("gen_add_toc", True)
    ss.setdefault("gen_tpl_has_headings", True)
    ss.setdefault("out_draft_bytes", b"")
    ss.setdefault("out_recs_bytes", b"")
    ss.setdefault("dyn_recos_preview", {})
    ss.setdefault("last_generation_meta", {})
    ss.setdefault("pending_blueprint", None)


def main():
    st.set_page_config(page_title="Proposal Studio", page_icon="📄", layout="wide")
    ensure_dirs(BASE_DIR)

    _init_session_defaults()

    cfg = Config()
    oai = OpenAI()

    # Sidebar navigation
    st.sidebar.title("Proposal Studio")
    page = st.sidebar.radio(
        "Navigate",
        ["Home", "Understanding", "Chats", "Generation", "Settings"],
        index=0,
    )

    if page == "Home":
        render_home(cfg, oai)
    elif page == "Understanding":
        render_understanding(cfg, oai, RFP_FACTS_DIR, SUPPORTED_UPLOAD_TYPES, TEMPLATE_DIR, STATIC_DIR, STATIC_MAP_FILE, STATIC_SECTIONS)
    elif page == "Chats":
        render_sharepoint_chat(cfg, oai)
    elif page == "Generation":
        render_generation(cfg, oai, BASE_DIR, STATIC_DIR, TEMPLATE_DIR, STATIC_MAP_FILE, STATIC_SECTIONS)
    elif page == "Settings":
        render_settings(cfg, BASE_DIR=BASE_DIR, TEMPLATE_DIR=TEMPLATE_DIR, STATIC_DIR=STATIC_DIR, STATIC_MAP_FILE=STATIC_MAP_FILE, STATIC_SECTIONS=STATIC_SECTIONS)


if __name__ == "__main__":
    main()
