# main.py
# Streamlit front-end for Azure-friendly Proposal Generator
# - No nested expanders (uses safe_expander + tabs)
# - SQLite preflight for Azure Web App sanity
# - Defensive imports so the UI keeps working during refactors
# - Conditional visibility of "Generate" buttons after Q&A submission

from __future__ import annotations
import io
import os
import sys
import json
import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# ----------------------------
# Preflight: sqlite & platform
# ----------------------------
def preflight_sqlite():
    try:
        import sqlite3, platform
        st.session_state["_sqlite_ok"] = True
        st.session_state["_sqlite_info"] = {
            "python_version": platform.python_version(),
            "sqlite_version": sqlite3.sqlite_version,
            "library": getattr(sqlite3, "sqlite_version_info", None),
        }
        # Log to server console too
        print(
            "SQLite OK | Python:",
            platform.python_version(),
            "| SQLite lib ver:",
            sqlite3.sqlite_version,
        )
    except Exception as e:
        st.session_state["_sqlite_ok"] = False
        st.session_state["_sqlite_info"] = {"error": str(e)}
        print("SQLite import failed:", e)

preflight_sqlite()

# ----------------------------
# Optional / defensive imports
# ----------------------------
def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None

utils = _try_import("utils")
app_helpers = _try_import("app_helpers")
text_extraction = _try_import("text_extraction")
llm_processing = _try_import("llm_processing")
understanding_extractor = _try_import("understanding_extractor")
proposal_builder = _try_import("proposal_builder")
docx_generator = _try_import("docx_generator")
docx_assembler = _try_import("docx_assembler")
understanding_ppt_generator = _try_import("understanding_ppt_generator")

sp_retrieval = _try_import("sp_retrieval")
sharepoint = _try_import("sharepoint")
vectorstore = _try_import("vectorstore")
static_manager = _try_import("static_manager")
sp_sources = _try_import("sp_sources")
url_store = _try_import("url_store")
ui_helpers = _try_import("ui_helpers")

# ----------------------------
# Expander safety: never nest
# ----------------------------
@contextmanager
def safe_expander(label: str, **kwargs):
    """
    Streamlit forbids nesting expanders. This context-manager converts any
    nested expander into a labeled container automatically.
    """
    depth = st.session_state.get("_exp_depth", 0)
    if depth > 0:
        st.markdown(f"### {label}")
        with st.container():
            yield
    else:
        st.session_state["_exp_depth"] = depth + 1
        try:
            with st.expander(label, **kwargs):
                yield
        finally:
            st.session_state["_exp_depth"] = depth

# ----------------------------
# Session init
# ----------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("sp_cfg", {"site_url": "", "include_exts": [".pdf", ".docx", ".txt"]})
    ss.setdefault("sp_ingested", False)
    ss.setdefault("uploaded_docs", [])      # List[Dict{name, type, bytes}]
    ss.setdefault("rfp_texts", [])          # List[str]
    ss.setdefault("understanding", {})      # raw chunks/points from extractor
    ss.setdefault("brief", None)            # str/markdown
    ss.setdefault("scope", None)            # str/markdown
    ss.setdefault("intel_questions", [])    # List[{"q": str, "options": [str], "multi": bool}]
    ss.setdefault("intel_answers", {})      # Dict[q] -> selection(s)
    ss.setdefault("qa_submitted", False)
    ss.setdefault("doc_ready", False)
    ss.setdefault("_exp_depth", 0)

init_state()

# ----------------------------
# Helpers
# ----------------------------
def show_status(label: str, ok: bool, detail: Optional[str] = None):
    col1, col2 = st.columns([1, 5])
    with col1:
        st.success("OK") if ok else st.warning("Check")
    with col2:
        st.write(f"**{label}**")
        if detail:
            st.caption(detail)

def extract_text_from_uploads(files: List[Dict[str, Any]]) -> List[str]:
    texts: List[str] = []
    for f in files:
        name = f["name"].lower()
        data = f["data"]
        try:
            if text_extraction:
                # Prefer your project’s text extractor if present
                txt = text_extraction.extract_text_bytes(name, data)
                if txt:
                    texts.append(txt)
                    continue
            # Fallback: naive decoding for .txt
            if name.endswith(".txt"):
                texts.append(data.decode("utf-8", errors="ignore"))
            else:
                st.info(f"Using fallback for {f['name']} (consider enabling text_extraction.py)")
        except Exception as e:
            st.error(f"Failed to read {f['name']}: {e}")
    return texts

def run_understanding(rfp_texts: List[str]) -> Tuple[Dict, str, str, List[Dict]]:
    """
    Returns: (understanding_dict, brief_md, scope_md, intelligence_questions)
    intelligence_questions format:
      [{"q": "...", "options": ["A","B","C"], "multi": False}, ...]
    """
    understanding = {}
    brief, scope = None, None
    questions: List[Dict] = []

    try:
        if understanding_extractor and hasattr(understanding_extractor, "extract"):
            understanding = understanding_extractor.extract(rfp_texts)
    except Exception as e:
        st.warning(f"understanding_extractor.extract() error: {e}")

    try:
        if llm_processing and hasattr(llm_processing, "generate_proposal_brief"):
            brief = llm_processing.generate_proposal_brief(rfp_texts, understanding)
        if llm_processing and hasattr(llm_processing, "generate_scope_snapshot"):
            scope = llm_processing.generate_scope_snapshot(rfp_texts, understanding)
        if llm_processing and hasattr(llm_processing, "generate_intelligence_questions"):
            questions = llm_processing.generate_intelligence_questions(rfp_texts, understanding)
    except Exception as e:
        st.warning(f"LLM processing error: {e}")

    # Safe fallback content
    if not brief:
        brief = "*(Auto-brief placeholder)* Summary of the RFP will be generated here."
    if not scope:
        scope = "*(Auto-scope placeholder)* Key deliverables, assumptions, exclusions, and SLAs."
    if not questions:
        questions = [
            {"q": "Engagement model?", "options": ["Fixed Price", "T&M", "Hybrid"], "multi": False},
            {"q": "Hosting preference?", "options": ["Azure", "AWS", "On-Prem"], "multi": False},
            {"q": "Data residency constraints?", "options": ["EU", "US", "India", "No preference"], "multi": True},
        ]

    return understanding, brief, scope, questions

def can_generate_docs() -> bool:
    return bool(st.session_state.get("qa_submitted") and st.session_state.get("brief") and st.session_state.get("scope"))

def safe_generate_docx():
    try:
        if docx_generator and hasattr(docx_generator, "generate"):
            return docx_generator.generate(
                brief=st.session_state["brief"],
                scope=st.session_state["scope"],
                understanding=st.session_state["understanding"],
                qa=st.session_state["intel_answers"],
            )
        if proposal_builder and hasattr(proposal_builder, "build_proposal_docx"):
            return proposal_builder.build_proposal_docx(
                brief=st.session_state["brief"],
                scope=st.session_state["scope"],
                understanding=st.session_state["understanding"],
                qa=st.session_state["intel_answers"],
            )
    except Exception as e:
        st.error(f"DOCX generation failed: {e}")
    # Fallback: create a simple .txt and tell the user to export via your pipeline
    content = "\n\n".join(
        [
            "# Executive Brief",
            st.session_state["brief"] or "",
            "# Scope Snapshot",
            st.session_state["scope"] or "",
            "# Intelligence Answers",
            json.dumps(st.session_state["intel_answers"], indent=2),
        ]
    ).encode("utf-8")
    return ("proposal.txt", content, "text/plain")

def safe_generate_ppt():
    try:
        if understanding_ppt_generator and hasattr(understanding_ppt_generator, "generate"):
            return understanding_ppt_generator.generate(
                brief=st.session_state["brief"],
                scope=st.session_state["scope"],
                understanding=st.session_state["understanding"],
                qa=st.session_state["intel_answers"],
            )
    except Exception as e:
        st.error(f"PPT generation failed: {e}")
    # Fallback: none; return None indicates we couldn’t build a PPT
    return None

def ingest_sharepoint(cfg: Dict[str, Any], include_exts: List[str]) -> Tuple[bool, str]:
    """
    Try to ingest SharePoint content into your vector store.
    """
    try:
        if sp_retrieval and hasattr(sp_retrieval, "ingest_sharepoint"):
            ok, msg = sp_retrieval.ingest_sharepoint(cfg, include_exts=include_exts)
            return (ok, msg or "Ingest complete.")
        # Legacy path seen in older traces
        if vectorstore and hasattr(vectorstore, "ingest_sharepoint"):
            ok = vectorstore.ingest_sharepoint(cfg, include_exts=include_exts)
            return (bool(ok), "Vectorstore SharePoint ingest complete." if ok else "Vectorstore ingest returned False.")
    except Exception as e:
        return (False, f"Ingest error: {e}")
    return (False, "SharePoint modules not available; skipping.")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    # SharePoint config
    st.subheader("SharePoint")
    site_url = st.text_input("Site URL", value=st.session_state["sp_cfg"].get("site_url", ""))
    exts = st.multiselect(
        "Include file types",
        options=[".pdf", ".docx", ".txt", ".pptx", ".xlsx"],
        default=st.session_state["sp_cfg"].get("include_exts", [".pdf", ".docx", ".txt"]),
    )
    col_ing1, col_ing2 = st.columns([1, 1])
    with col_ing1:
        if st.button("Ingest SharePoint", use_container_width=True):
            st.session_state["sp_cfg"]["site_url"] = site_url
            st.session_state["sp_cfg"]["include_exts"] = exts
            with st.spinner("Ingesting from SharePoint…"):
                ok, msg = ingest_sharepoint(st.session_state["sp_cfg"], exts)
            st.session_state["sp_ingested"] = ok
            st.toast("SharePoint ingested." if ok else "SharePoint ingest failed.", icon="✅" if ok else "⚠️")
            if msg:
                st.caption(msg)
    with col_ing2:
        if st.button("Reset Ingest Flag", use_container_width=True):
            st.session_state["sp_ingested"] = False
            st.toast("Ingest flag cleared.", icon="🧹")

    st.divider()

    # File upload
    st.subheader("RFP Documents")
    up_files = st.file_uploader(
        "Upload RFPs (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if up_files:
        # Persist uploaded docs
        stored = []
        for f in up_files:
            stored.append({"name": f.name, "type": f.type, "data": f.read()})
        st.session_state["uploaded_docs"] = stored
        st.toast(f"{len(stored)} file(s) staged.", icon="📄")

    st.divider()
    st.caption("Azure check:")
    show_status(
        "SQLite available",
        bool(st.session_state.get("_sqlite_ok")),
        detail=str(st.session_state.get("_sqlite_info")),
    )

# ----------------------------
# Main UI
# ----------------------------
st.set_page_config(page_title="Proposal Generator", layout="wide")
st.title("📄 Proposal Generator (Azure-friendly)")

# Section: Data Sources (uses a single expander; sub-sections use containers/tabs)
with safe_expander("1) Data Sources", expanded=True):
    cols = st.columns(2)
    with cols[0]:
        st.subheader("SharePoint")
        st.write("Status:")
        show_status(
            "Ingested into Vector Store",
            bool(st.session_state.get("sp_ingested")),
            detail=("Using project modules" if sp_retrieval or vectorstore else "Vector store modules not found"),
        )
        if st.session_state["sp_cfg"].get("site_url"):
            st.caption(f"Site: {st.session_state['sp_cfg']['site_url']}")
        st.caption(f"Included types: {', '.join(st.session_state['sp_cfg'].get('include_exts', []))}")

    with cols[1]:
        st.subheader("Uploaded RFPs")
        if st.session_state["uploaded_docs"]:
            st.write(f"{len(st.session_state['uploaded_docs'])} file(s) ready:")
            for f in st.session_state["uploaded_docs"]:
                st.markdown(f"- {f['name']}")
        else:
            st.info("Upload one or more RFP documents from the sidebar.")

# Section: Understanding
with safe_expander("2) Proposal Understanding", expanded=True):
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("▶️ Run Understanding", use_container_width=True, help="Extract brief, scope, and questions"):
            if not st.session_state["uploaded_docs"]:
                st.warning("Please upload at least one RFP document.")
            else:
                with st.spinner("Extracting and summarizing…"):
                    texts = extract_text_from_uploads(st.session_state["uploaded_docs"])
                    st.session_state["rfp_texts"] = texts
                    understanding, brief, scope, qs = run_understanding(texts)
                    st.session_state["understanding"] = understanding
                    st.session_state["brief"] = brief
                    st.session_state["scope"] = scope
                    st.session_state["intel_questions"] = qs
                    st.session_state["intel_answers"] = {}
                    st.session_state["qa_submitted"] = False
                st.toast("Understanding ready.", icon="✅")
    with c2:
        st.caption("Use this step to transform raw RFP text into a working brief, a scope outline, and targeted clarification questions.")

    tabs = st.tabs(["RFP Summary", "Scope Snapshot", "Intelligence Q&A"])

    with tabs[0]:
        if st.session_state["brief"]:
            st.markdown(st.session_state["brief"])
        else:
            st.info("Click **Run Understanding** to generate the summary.")

    with tabs[1]:
        if st.session_state["scope"]:
            st.markdown(st.session_state["scope"])
        else:
            st.info("Click **Run Understanding** to generate the scope snapshot.")

    with tabs[2]:
        qs = st.session_state.get("intel_questions", [])
        if not qs:
            st.info("No questions yet. Click **Run Understanding** first.")
        else:
            st.write("Please select answers and submit:")
            answers = {}
            for i, q in enumerate(qs, start=1):
                key = f"q_{i}"
                label = f"{i}. {q.get('q','')}"
                options = q.get("options", [])
                multi = bool(q.get("multi", False))
                if multi:
                    answers[label] = st.multiselect(label, options=options, key=key)
                else:
                    answers[label] = st.radio(label, options=options, key=key, horizontal=True)
            if st.button("✅ Submit Answers", type="primary"):
                st.session_state["intel_answers"] = answers
                st.session_state["qa_submitted"] = True
                st.session_state["doc_ready"] = can_generate_docs()
                st.success("Answers captured. You can now proceed to document generation.")

# Section: Document Generation
with safe_expander("3) Generate Deliverables", expanded=True):
    disabled = not can_generate_docs()
    if disabled:
        st.info("Provide Q&A answers (and ensure brief/scope are generated) to enable exports.")

    tabs_out = st.tabs(["DOCX", "PPT"])
    with tabs_out[0]:
        st.write("Create a proposal document.")
        colg1, colg2 = st.columns([1, 3])
        with colg1:
            if st.button("📝 Generate DOCX", disabled=disabled, use_container_width=True):
                with st.spinner("Building DOCX…"):
                    result = safe_generate_docx()
                if isinstance(result, tuple) and len(result) == 3:
                    fname, data, mime = result
                    st.download_button("Download", data=data, file_name=fname, mime=mime, use_container_width=True)
                    st.success("DOCX ready.")
                else:
                    # Expecting your generator to return (file_name, bytes, mime)
                    st.warning("The DOCX generator didn’t return a file tuple; please check generator module.")
        with colg2:
            st.caption("The DOCX includes the executive brief, scope snapshot, and your Q&A selections. You can wire your own Word template engine via `docx_generator` / `proposal_builder`.")

    with tabs_out[1]:
        st.write("Create a presentation.")
        colp1, colp2 = st.columns([1, 3])
        with colp1:
            if st.button("📊 Generate PPT", disabled=disabled, use_container_width=True):
                with st.spinner("Building PPT…"):
                    result = safe_generate_ppt()
                if isinstance(result, tuple) and len(result) == 3:
                    fname, data, mime = result
                    st.download_button("Download", data=data, file_name=fname, mime=mime, use_container_width=True)
                    st.success("PPT ready.")
                else:
                    st.warning("PPT generator not available or returned no file. Hook up `understanding_ppt_generator.generate`.")
        with colp2:
            st.caption("Slides will summarize the brief, scope, and key choices from Q&A.")

# Section: Diagnostics (no nested expanders)
with safe_expander("Diagnostics", expanded=False):
    env_cols = st.columns(3)
    with env_cols[0]:
        st.write("**Python & SQLite**")
        st.json(st.session_state.get("_sqlite_info", {}))
    with env_cols[1]:
        st.write("**SharePoint**")
        st.json(st.session_state.get("sp_cfg", {}))
        st.write("Ingested:", st.session_state.get("sp_ingested"))
    with env_cols[2]:
        st.write("**State Bits**")
        st.write("Uploads:", len(st.session_state.get("uploaded_docs", [])))
        st.write("Text parts:", len(st.session_state.get("rfp_texts", [])))
        st.write("Q&A submitted:", bool(st.session_state.get("qa_submitted")))
        st.write("Doc ready:", bool(st.session_state.get("doc_ready")))

