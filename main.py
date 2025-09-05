# main.py
# -----------------------------------------------------------------------------
# Adds "Smart Retrieval (evidence packs)" to:
#  - Chat (SharePoint DB)  ← ChatGPT-like with streaming & citations
#  - Chat with your RFP (Understanding tab)  ← also ChatGPT-like now
# Shows diagnostics (evidence snippets) per assistant message when available.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
from time import strftime, localtime
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
from openai import OpenAI



import textwrap
from config import load_config, Config
from modules.ui_helpers import (
    inject_sticky_chat_css, copy_to_clipboard_button, copy_sources_button,
    enable_enter_to_send, render_rfp_facts
)
from modules.app_helpers import (
    ensure_dirs, list_files, load_saved_urls, save_saved_urls, upsert_url, delete_url,
    load_static_map, save_static_map, load_sp_ingested_map, save_sp_ingested_map,
    st_rerun_compat, get_sp_docs_any, sp_index_stats, get_passages_for_query,
    rag_answer_uploaded, list_blueprints, save_blueprint, load_blueprint, filter_struct_questions,
    save_rfp_facts_json, list_rfp_facts_json, load_rfp_facts_json, insert_table_of_contents
)
from modules.utils import save_to_temp, cleanup_temp_files
from modules.vectorstore import init_uploaded_store, init_sharepoint_store, ingest_files, ingest_sharepoint, wipe_up_store, wipe_sp_store
from modules.sharepoint import canonicalize_site_url
from modules.text_extraction import extract_text, ocr_pdf
from modules.llm_processing import generate_intelligence_questions, generate_structuring_questions
from modules.understanding_extractor import extract_rfp_facts, extract_rfp_facts_from_raw_text, facts_to_context, plan_dynamic_sections, plan_to_blueprint
from modules.proposal_builder import BuildOptions, build_proposal
from modules.sp_retrieval import get_sp_evidence_for_question, get_store_evidence
from modules.understanding_docx import build_understanding_docx


# ---------------- basic page config / styles ----------------
st.set_page_config(page_title="BCT Proposal Studio", layout="wide", page_icon="📄")
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
.sticky { position: sticky; top: 0; z-index: 9; background: var(--background-color); padding: .4rem 0 .6rem 0; }
.badge { display:inline-block; padding:.15rem .5rem; border-radius:999px; font-size:.75rem; background:rgba(3,158,237,.12); color:#039eed; border:1px solid rgba(3,158,237,.25); }
section[data-testid="stChatMessage"] { margin-bottom: .35rem; }
.smallcap { font-size:.8rem; opacity:.8; }
</style>
""", unsafe_allow_html=True)

st.title("BCT Proposal Studio")
cfg = load_config()
oai = OpenAI(api_key=cfg.OPENAI_API_KEY)

# ----------------- constants/paths -----------------
BASE_DIR = Path(getattr(cfg, "BASE_DIR", "."))
SP_URL_STORE = BASE_DIR / "sp_urls.json"
SP_INGEST_MAP_FILE = BASE_DIR / "sp_ingested_map.json"
STATIC_DIR = BASE_DIR / "static_sources"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_MAP_FILE = BASE_DIR / "static_map.json"
RFP_FACTS_DIR = BASE_DIR / "rfp_facts"
SUPPORTED_UPLOAD_TYPES = ["pdf", "docx", "pptx"]

STATIC_SECTIONS = ["Profile of the Firm","Cover Letter","Executive Summary","Experience","Offerings","References","Team & Credentials","Case Studies","Confidentiality"]

import numpy as _np
if not hasattr(_np, "float_"): _np.float_ = _np.float64
if not hasattr(_np, "int_"): _np.int_ = _np.int64
if not hasattr(_np, "complex_"): _np.complex_ = _np.complex128


try:
    import sqlite3 as _stdlib_sqlite
    def _vtuple(v: str):
        try:
            return tuple(int(x) for x in v.split(".")[:3])
        except Exception:
            return (0, 0, 0)
    if _vtuple(getattr(_stdlib_sqlite, "sqlite_version", "0.0.0")) < (3, 35, 0):
        import sys
        import pysqlite3 as _pysqlite3  # requires pysqlite3-binary on Linux
        sys.modules["sqlite3"] = _pysqlite3
        import sqlite3
        print("SQLite patched via pysqlite3:", sqlite3.sqlite_version)
    else:
        print("System SQLite OK:", _stdlib_sqlite.sqlite_version)
except Exception as _e:
    print("SQLite backport not applied:", _e)


# ----------------- session state -----------------
ss = st.session_state
ss.setdefault("uploaded_paths", []); ss.setdefault("up_store", None); ss.setdefault("vectorized", False); ss.setdefault("temp_files", [])
ss.setdefault("sp_ingested_files", []); ss.setdefault("sp_ingested_map", load_sp_ingested_map(SP_INGEST_MAP_FILE)); ss.setdefault("sp_urls", load_saved_urls(SP_URL_STORE)); ss.setdefault("sp_selected_idx", 0)
ss.setdefault("rfp_facts", None); ss.setdefault("rfp_raw", ""); ss.setdefault("section_plan", [])
ss.setdefault("ocr_enable", True); ss.setdefault("ocr_used_last", False); ss.setdefault("raw_chars_last", 0)
ss.setdefault("dyn_recos_preview", {}); ss.setdefault("out_draft_bytes", None); ss.setdefault("out_recs_bytes", None); ss.setdefault("last_generation_meta", {})
ss.setdefault("gen_tpl", "— choose a template —"); ss.setdefault("gen_static_sel", []); ss.setdefault("gen_dyn_sel", [])
ss.setdefault("gen_use_anchors", True); ss.setdefault("gen_add_headings", False); ss.setdefault("gen_include_sources", True)
ss.setdefault("gen_top_k", 6);  
ss.setdefault("gen_page_breaks", True); ss.setdefault("gen_add_toc", True); ss.setdefault("gen_tpl_has_headings", True)
ss.setdefault("pending_blueprint", None)

ss.setdefault("gen_rec_style", "paragraphs")      # was "bullets"
ss.setdefault("gen_per_section_k", 25)            # was 0

# (chat states for SP & RFP chats)
ss.setdefault("sp_chat_messages", []); ss.setdefault("sp_chat_last_evidence", [])
ss.setdefault("rfp_chat_messages", []); ss.setdefault("rfp_chat_last_evidence", [])

# ---------------- small helpers for evidence rendering ----------------
def _get(ev, key, default=None):
    if isinstance(ev, dict):
        return ev.get(key, default)
    return getattr(ev, key, default)

def _fmt_why(ev) -> str:
    v = _get(ev, "why_relevant")
    return f" — {v}" if v else ""

def _fmt_src(ev) -> str:
    src = _get(ev, "source", "")
    page = _get(ev, "page_hint", "")
    return f"{src} {page}".strip() if src or page else ""

def _fmt_text(ev) -> str:
    return _get(ev, "text", "") or ""
#-------------------------For Understanding Doc-------------------

# --- VALUE PROP synthesis (uses SOW + SharePoint + LLM) ---


def _make_bct_value_props(
    cfg, oai, facts: dict, k: int = 25, *, lenient: bool = True, temperature: float = 0.25
) -> tuple[str, list[str], str]:
    """Return (bct_vp_text, bct_vp_sources_labels, generic_vp_text)."""

    # ---- 0) Inputs & clamps
    k = max(8, min(50, int(k or 25)))  # keep evidence pack in a sane range

    # ---- 1) Get SOW (fallback to facts summary)
    sow = ((facts or {}).get("contract_and_compliance") or {}).get("sow_summary", "").strip()
    if not sow:
        try:
            from modules.understanding_extractor import facts_to_context
            sow = (facts_to_context(facts).summary or "").strip()
        except Exception:
            sow = ""
    sow_snip = sow[:1500]  # keep prompt lean

    # ---- 2) Add light cues from facts to steer retrieval
    sol = (facts or {}).get("solicitation", {}) or {}
    org = (facts or {}).get("proposal_organization", {}) or {}
    evalc = (facts or {}).get("evaluation_and_selection", {}) or {}
    cues = []
    if sol.get("issuing_entity_name"): cues.append(sol["issuing_entity_name"])
    if sol.get("solicitation_title"):  cues.append(sol["solicitation_title"])
    if org.get("required_sections"):   cues += org["required_sections"][:5]
    if evalc.get("criteria"):
        cues += [c.get("name","") for c in evalc["criteria"] if c.get("name")]
    cues = [c for c in cues if c]
    cue_line = ("Keywords: " + ", ".join(sorted(set(cues))[:10])) if cues else ""

    # ---- 3) Build query for SP-grounded evidence
    query = textwrap.dedent(f"""\
        From BCT's perspective, what compelling value proposition aligns with this SOW?
        Focus on differentiators, past performance, accelerators/solutions, certifications,
        delivery approach, governance/SLAs, security/compliance, and measurable outcomes.

        {cue_line}

        SOW SUMMARY:
        {sow_snip}
    """).strip()

    # ---- 4) Retrieve evidence (lenient if supported)
    try:
        evs = get_sp_evidence_for_question(query, facts, cfg, k=k, lenient=lenient) or []
    except TypeError:
        evs = get_sp_evidence_for_question(query, facts, cfg, k=k) or []

    # ---- 5) Build numbered evidence block (stable index by (src,page))
    def _format_sources(evs):
        seen = set(); out = []
        for e in evs:
            src = getattr(e, "source", None) or (e.get("source") if isinstance(e, dict) else None)
            pg  = getattr(e, "page_hint", None) or (e.get("page_hint") if isinstance(e, dict) else None)
            if not src: 
                continue
            label = f"{Path(src).name}{f' · p.{pg}' if pg else ''}"
            key = f"{src}|{pg or ''}"
            if key not in seen:
                seen.add(key)
                out.append(label)
        return out

    def _build_context_block(evs, max_chunks=24):
        numbered, src_map, idx_ctr = [], {}, 1
        for e in evs:
            src = getattr(e, "source", None) or (e.get("source") if isinstance(e, dict) else None)
            pg  = getattr(e, "page_hint", None) or (e.get("page_hint") if isinstance(e, dict) else None)
            txt = getattr(e, "text", None) or (e.get("text") if isinstance(e, dict) else "")
            key = f"{src}|{pg or ''}"
            if key not in src_map:
                src_map[key] = idx_ctr; idx_ctr += 1
            if len(numbered) < max_chunks and txt:
                numbered.append(f"[{src_map[key]}] {txt}")
        return "\n\n".join(numbered)

    context_block = _build_context_block(evs)

    # ---- 6) Compose BCT VP (grounded if evidence, fallback to non-cited if none)
    if context_block:
        sys = (
            "You are a proposal strategist. Using only the provided evidence, craft a concise value proposition for BCT "
            "aligned to the SOW. Use markdown; prefer 2 short paragraphs (or 5–8 bullets). Append [n] footnotes where "
            "claims come from evidence [n]. If evidence is thin, say so briefly."
        )
        user = f"EVIDENCE (numbered):\n{context_block}\n\nTASK: Write the BCT value proposition."
        temp = temperature
    else:
        # No evidence—write a short evergreen VP tailored to the SOW (no citations)
        sys = (
            "You are a proposal strategist. Draft a short, evergreen BCT value proposition tailored to the SOW. "
            "Use plain markdown paragraphs. Do not invent client names or cite sources."
        )
        user = f"SOW SUMMARY:\n{sow_snip}\n\nTASK: Write the BCT value proposition (no citations)."
        temp = max(0.3, temperature)

    try:
        r = oai.chat.completions.create(
            model=getattr(cfg, "ANALYSIS_MODEL", "gpt-4o-mini"),
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=temp,
        )
        bct_vp_text = (r.choices[0].message.content or "").strip()
    except Exception as ex:
        bct_vp_text = f"*LLM error composing BCT value proposition: {ex}*"

    bct_vp_sources = _format_sources(evs)

    # ---- 7) Generic value prop (no citations)
    facts_slim = facts.get("solicitation", {}) if isinstance(facts, dict) else {}
    issuer = (facts_slim or {}).get("issuing_entity_name", "")
    title  = (facts_slim or {}).get("solicitation_title", "") or (facts_slim or {}).get("solicitation_id", "")
    hint_lines = []
    if title:  hint_lines.append(f"RFP Title: {title}")
    if issuer: hint_lines.append(f"Issuer: {issuer}")
    if sow:    hint_lines.append(f"SOW Summary: {sow_snip}")
    hint = "\n".join(hint_lines)

    sys2 = (
        "You are a proposal writer. Draft a reusable, generic value proposition (no citations), tailored to the hint. "
        "Use 1–2 plain markdown paragraphs (120–180 words). Avoid naming specific past clients; keep it evergreen."
    )
    user2 = f"HINT:\n{hint or '(no hint)'}\n\nTASK: Write a generic value proposition suitable for an RFP response."

    try:
        r2 = oai.chat.completions.create(
            model=getattr(cfg, "ANALYSIS_MODEL", "gpt-4o-mini"),
            messages=[{"role":"system","content":sys2},{"role":"user","content":user2}],
            temperature=0.4,
        )
        generic_vp_text = (r2.choices[0].message.content or "").strip()
    except Exception as ex:
        generic_vp_text = f"*LLM error composing generic value proposition: {ex}*"

    return bct_vp_text, bct_vp_sources, generic_vp_text


# ---------------- shared chat helpers (both SP & RFP) ----------------
def _rewrite_query_with_history(oai: OpenAI, cfg, history: List[Dict[str, str]], user_q: str) -> str:
    """Make a follow-up standalone using the last ~6 turns."""
    try:
        hist = history[-6:]
        hist_txt = "\n".join(f"{m['role']}: {m['content']}" for m in hist)
        prompt = (
            "Rewrite the user's last question so it is standalone and self-contained, "
            "preserving intent and nouns. Return ONLY the rewritten question.\n\n"
            f"CHAT HISTORY (latest last):\n{hist_txt}\n\n"
            f"USER QUESTION:\n{user_q}"
        )
        r = oai.chat.completions.create(
            model=getattr(cfg, "ANALYSIS_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=128,
        )
        out = (r.choices[0].message.content or "").strip()
        return out or user_q
    except Exception:
        return user_q

def _format_sources(evs: List[Any]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """Unique ordered sources + mapping path|page → [index]."""
    seen = {}
    sources: List[Dict[str, str]] = []
    for e in evs:
        src = getattr(e, "source", None) or (e.get("source") if isinstance(e, dict) else None)
        pg  = getattr(e, "page_hint", None) or (e.get("page_hint") if isinstance(e, dict) else None)
        if not src:
            continue
        label = f"{Path(src).name}{f' · p.{pg}' if pg else ''}"
        key = f"{src}|{pg or ''}"
        if key not in seen:
            seen[key] = len(seen) + 1
            sources.append({"label": label, "path": src})
    return sources, seen

def _build_context_block(evs: List[Any], max_chunks: int = 18) -> str:
    """Numbered context aligned with sources for footnotes [n]."""
    sources, index_map = _format_sources(evs)
    numbered = []
    used = 0
    for e in evs:
        if used >= max_chunks:
            break
        src = getattr(e, "source", None) or (e.get("source") if isinstance(e, dict) else None)
        pg  = getattr(e, "page_hint", None) or (e.get("page_hint") if isinstance(e, dict) else None)
        txt = getattr(e, "text", None) or (e.get("text") if isinstance(e, dict) else "")
        key = f"{src}|{pg or ''}"
        idx = index_map.get(key)
        if not idx:
            continue
        numbered.append(f"[{idx}] {txt}")
        used += 1
    return "\n\n".join(numbered)

def _render_sources_ui(evs: List[Any]):
    sources, _ = _format_sources(evs)
    if not sources:
        return
    with st.expander("Sources", expanded=True):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**[{i}]** {s['label']}")

# ---------------- SharePoint Chat (ChatGPT-like) ----------------
def render_sharepoint_chat(cfg, oai):
    st.subheader("Chat with SharePoint Vector DB")
    st.caption("Grounded answers with citations. Ask follow-ups naturally.")
    inject_sticky_chat_css()

    # Settings row
    with st.container():
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        with c1:
            top_k = st.slider("Top-K", 4, 24, 12, 1, help="How many chunks to retrieve")
        with c2:
            temperature = st.slider("Creativity", 0.0, 1.0, 0.2, 0.1)
        with c3:
            lenient = st.toggle("Lenient retrieval", value=True, help="Wider, recall-oriented search")
        with c4:
            if st.button("Reset chat", use_container_width=True):
                ss.sp_chat_messages = []
                ss.sp_chat_last_evidence = []
                st.rerun()

    # Index status
    with st.expander("Index status & debug", expanded=False):
        chunks, vec_path = sp_index_stats(cfg)
        st.write(f"SharePoint index path: `{vec_path}`")
        st.write(f"Chunks available: **{chunks}**")
        if chunks == 0:
            st.info("The index appears empty. Use Settings → SharePoint → Pull & Index.")

    # History
    for m in ss.sp_chat_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Input
    user_q = st.chat_input("Ask SharePoint…")
    if not user_q:
        st.caption("Try: “submission due date for SOC RFP?”, “who’s the POC for …?”, “evaluation criteria weights”.")
        return

    # Append user message
    ss.sp_chat_messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # Rewrite → standalone
    standalone_q = _rewrite_query_with_history(oai, cfg, ss.sp_chat_messages, user_q)

    # Retrieve evidence
    k = top_k if lenient else max(10, min(25, top_k // 2))
    evs = get_sp_evidence_for_question(standalone_q, ss.get("rfp_facts"), cfg, k=k) or []
    ss.sp_chat_last_evidence = evs
    context_block = _build_context_block(evs)

    # Compose prompts
    sys = (
        "You are a precise assistant for SharePoint documents. Answer ONLY using the provided context. "
        "Cite sources with footnotes like [1], [2], aligned to the Sources list. If not found, say so."
    )
    user = (
        f"CONTEXT (numbered excerpts):\n{context_block or '[no evidence retrieved]'}\n\n"
        f"QUESTION:\n{standalone_q}\n\n"
        "Answer clearly in markdown. When you use a snippet, append the matching [n] footnote."
    )

    # Stream the answer
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""
        try:
            stream = oai.chat.completions.create(
                model=getattr(cfg, "ANALYSIS_MODEL", "gpt-4o"),
                messages=[{"role":"system","content":sys},{"role":"user","content":user}],
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                # OpenAI SDK v1: delta.content (guarded)
                token = ""
                try:
                    token = chunk.choices[0].delta.content or ""
                except Exception:
                    pass
                if token:
                    full += token
                    placeholder.markdown(full)
        except Exception as ex:
            full = f"*LLM error: {ex}*"
            placeholder.markdown(full)

        ss.sp_chat_messages.append({"role": "assistant", "content": full})
        _render_sources_ui(evs)

        # Quick follow-ups
        if full and len(full) > 40:
            cols = st.columns(3)
            suggs = ["Show key dates as bullets", "Who is the primary POC?", "Any page limits or forms?"]
            for i, s in enumerate(suggs):
                with cols[i]:
                    if st.button(s, use_container_width=True):
                        ss.sp_chat_messages.append({"role":"user","content":s})
                        st.rerun()

# ---------------- RFP Chat (ChatGPT-like) ----------------
def render_rfp_chat(cfg, oai):
    """Chat with the uploaded RFP. Supports:
       - Grounded (retrieval + citations)
       - Open LLM (no retrieval, no citations)
    """
    st.markdown("### Ask Anything (Chat with your RFP)")
    inject_sticky_chat_css()

    # Controls
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        mode = st.radio(
            "Mode",
            options=["Grounded (with citations)", "Open LLM (no citations)"],
            index=0,
            horizontal=False,
            key="rfp_mode",
            help="Grounded uses retrieved evidence with [n] footnotes; Open LLM is a free-form assistant."
        )
    with c2:
        top_k = st.slider("Top-K (RFP)", 4, 24, 12, 1, key="rfp_top_k", help="Used only in Grounded mode")
    with c3:
        temperature = st.slider("Creativity", 0.0, 1.0, 0.2, 0.1, key="rfp_temp")
    with c4:
        lenient = st.toggle("Lenient retrieval", value=True, key="rfp_lenient")

    if st.button("Reset RFP chat", use_container_width=True):
        ss.rfp_chat_messages = []
        ss.rfp_chat_last_evidence = []
        st.rerun()

    # History
    for m in ss.rfp_chat_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Input
    user_q = st.chat_input("Ask about the uploaded documents…")
    if not user_q:
        if mode.startswith("Grounded"):
            st.caption("Try: “what’s the submission due date?”, “POC details”, “evaluation criteria & weights”.")
        else:
            st.caption("Open mode: ask planning, drafting, or brainstorming questions (no citations).")
        return

    # Log user turn
    ss.rfp_chat_messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # Make follow-ups standalone
    standalone_q = _rewrite_query_with_history(oai, cfg, ss.rfp_chat_messages, user_q)

    if mode.startswith("Grounded"):
        # ------------------ GROUNDED PATH: retrieve evidence + cite ------------------
        k = top_k if lenient else max(6, top_k // 2)
        evs = get_store_evidence(ss.up_store, standalone_q, ss.rfp_facts, k=k) if ss.up_store else []
        ss.rfp_chat_last_evidence = evs
        context_block = _build_context_block(evs)

        sys = (
            "You are a precise assistant for the user's uploaded RFP documents. "
            "Answer ONLY using the provided context. Cite sources with [1], [2]. "
            "If not found in context, say you don't know."
        )
        user = (
            f"CONTEXT (numbered excerpts):\n{context_block or '[no evidence retrieved]'}\n\n"
            f"QUESTION:\n{standalone_q}\n\n"
            "Answer clearly in markdown and footnote claims with [n] that match the Sources list."
        )

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""
            try:
                stream = oai.chat.completions.create(
                    model=getattr(cfg, "ANALYSIS_MODEL", "gpt-4o"),
                    messages=[{"role":"system","content":sys},{"role":"user","content":user}],
                    temperature=temperature,
                    stream=True,
                )
                for chunk in stream:
                    token = ""
                    try:
                        token = chunk.choices[0].delta.content or ""
                    except Exception:
                        pass
                    if token:
                        full += token
                        placeholder.markdown(full)
            except Exception as ex:
                full = f"*LLM error: {ex}*"
                placeholder.markdown(full)

            ss.rfp_chat_messages.append({"role": "assistant", "content": full})
            _render_sources_ui(evs)

    else:
        # ------------------ OPEN LLM PATH: no retrieval, no citations ------------------
        # Optional tiny hint from extracted facts (keeps answers on-topic without forcing citations)
        facts = ss.get("rfp_facts") or {}
        issuer = (facts.get("solicitation") or {}).get("issuing_entity_name") or ""
        title  = (facts.get("solicitation") or {}).get("solicitation_title") or (facts.get("solicitation") or {}).get("solicitation_id") or ""
        due    = ((facts.get("schedule") or {}).get("submission_due") or {}).get("date") or ""
        hint_lines = []
        if title:  hint_lines.append(f"RFP Title: {title}")
        if issuer: hint_lines.append(f"Issuer: {issuer}")
        if due:    hint_lines.append(f"Due: {due}")
        hint = "\n".join(hint_lines)

        sys = (
            "You are a helpful proposal assistant. Answer clearly in markdown. "
            "You are not required to cite sources. If the user asks for citations, say this is open mode."
        )
        user = (
            (f"RFP FACT HINTS (optional):\n{hint}\n\n" if hint else "") +
            f"USER QUESTION:\n{standalone_q}"
        )

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""
            try:
                stream = oai.chat.completions.create(
                    model=getattr(cfg, "ANALYSIS_MODEL", "gpt-4o"),
                    messages=[{"role":"system","content":sys},{"role":"user","content":user}],
                    temperature=temperature,
                    stream=True,
                )
                for chunk in stream:
                    token = ""
                    try:
                        token = chunk.choices[0].delta.content or ""
                    except Exception:
                        pass
                    if token:
                        full += token
                        placeholder.markdown(full)
            except Exception as ex:
                full = f"*LLM error: {ex}*"
                placeholder.markdown(full)

            ss.rfp_chat_messages.append({"role": "assistant", "content": full})
            # no Sources panel in open mode

# ================= ensure folders exist =================
ensure_dirs(BASE_DIR); RFP_FACTS_DIR.mkdir(parents=True, exist_ok=True)
SP_URL_STORE.parent.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- tabs -----------------
tab_chat_sp, tab_understanding, tab_generation, tab_settings = st.tabs(
    ["Chat (SharePoint DB)", "Proposal Understanding", "Proposal Generation", "Settings"]
)

# SQLite Sanity Check (debug print)
try:
    import sqlite3, platform
    print("SQLite OK | Python:", platform.python_version(), "| SQLite lib ver:", sqlite3.sqlite_version)
except Exception as e:
    print("SQLite import failed:", e)

# ================= Chat (SharePoint DB) =================
with tab_chat_sp:
    render_sharepoint_chat(cfg, oai)

    # Retrieval audit for last turn
    with st.expander("Show retrieved passages (last turn)", expanded=False):
        evs = ss.get("sp_chat_last_evidence") or []
        if evs:
            for j, e in enumerate(evs, start=1):
                st.markdown(f"**[{j}]** `{_fmt_src(e)}`{_fmt_why(e)}")
                st.code(_fmt_text(e)[:1200], language="text")
        else:
            st.caption("Ask a question to see retrieved passages here.")

# ================= Proposal Understanding =================
with tab_understanding:
    st.subheader("Proposal Understanding")
    st.markdown('<div class="sticky"><span class="badge">Step 1</span> Upload & Vectorize RFP</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Upload RFP Documents (multiple)", type=SUPPORTED_UPLOAD_TYPES, accept_multiple_files=True)

    if st.button("Generate RFP Analysis", type="primary"):
        new_files = []
        for uf in uploaded_files or []:
            if uf.name not in ss.uploaded_paths:
                path = save_to_temp(uf); new_files.append(path); ss.uploaded_paths.append(uf.name)
        if new_files:
            ss.up_store = init_uploaded_store(cfg)
            progress_text = st.empty(); progress_bar = st.progress(0.0); n_files = len(new_files)
            with st.spinner("Vectorizing uploaded documents…"):
                for i, path in enumerate(new_files, 1):
                    ingest_files([path], ss.up_store, getattr(cfg, "CHUNK_SIZE", 1000))
                    progress_bar.progress(i / n_files); progress_text.write(f"Processed {i} of {n_files} files")
            progress_bar.empty(); progress_text.empty()
            ss.vectorized = True; ss.temp_files.extend(new_files)
            ss.rfp_facts = None; ss.rfp_raw = ""; ss.section_plan = []
            # reset RFP chat on re-vectorize
            ss.rfp_chat_messages = []; ss.rfp_chat_last_evidence = []

    cc1, cc2, cc3 = st.columns([1,1,1])
    with cc1:
        if st.button("Clear Uploaded Vector Store"):
            wipe_up_store(cfg)
            ss.vectorized = False; ss.up_store = None; ss.uploaded_paths = []; ss.rfp_facts = None; ss.rfp_raw = ""; ss.section_plan = []
            ss.rfp_chat_messages = []; ss.rfp_chat_last_evidence = []
            st.success("Uploaded vector store cleared.")
    with cc2:
        if st.button("Cleanup uploaded temp files"):
            cleanup_temp_files(ss.temp_files); ss.temp_files = []; st.success("Temp files cleaned up.")
    with cc3:
        if ss.rfp_facts and st.button("Export facts JSON"):
            fname = f"rfp_facts_{strftime('%Y%m%d_%H%M%S', localtime())}.json"
            path = save_rfp_facts_json(RFP_FACTS_DIR, ss.rfp_facts, fname)
            st.success(f"Saved: {path}")

    st.markdown("### RFP Facts Summary")
    st.caption(f"Vectorized: {bool(ss.vectorized)} • Store ready: {ss.up_store is not None}")

    if ss.vectorized and ss.up_store is not None:
        # ---- Extract facts ----
        if ss.rfp_facts is None:
            with st.spinner("Extracting issuer, contacts, schedule, instructions, qualifications, and evaluation…"):
                try:
                    facts, raw = extract_rfp_facts(ss.up_store, oai, cfg, return_raw=True)
                except TypeError:
                    facts = extract_rfp_facts(ss.up_store, oai, cfg); raw = ""

                def _is_empty(d: dict) -> bool:
                    return not any(bool(d.get(k)) for k in [
                        "solicitation","points_of_contact","schedule","submission_instructions",
                        "minimum_qualifications","proposal_organization","evaluation_and_selection","contract_and_compliance"
                    ])

                ocr_used = False; chars = 0
                if not isinstance(facts, dict) or _is_empty(facts):
                    raw_blobs = []
                    for p in ss.temp_files:
                        try:
                            txt = extract_text(p) or ""
                            if txt.strip(): raw_blobs.append(txt)
                        except Exception:
                            continue
                    raw_text = "\n\n".join(raw_blobs); chars = len(raw_text)
                    if raw_text.strip():
                        with st.spinner("No RAG hits; retrying with raw-text fallback…"):
                            facts2, raw2 = extract_rfp_facts_from_raw_text(raw_text[:120000], oai, cfg, return_raw=True)
                            if isinstance(facts2, dict) and not _is_empty(facts2):
                                facts, raw = facts2, raw2

                if not isinstance(facts, dict) or not facts:
                    facts = {"missing_notes": ["Extractor returned empty JSON."]}

                ss.rfp_facts = facts; ss.rfp_raw = raw; ss.ocr_used_last = bool(ocr_used); ss.raw_chars_last = int(chars)

        render_rfp_facts(ss.rfp_facts or {}, show_provenance=True)

        # ---- Value Proposition (Preview) — shown on page BEFORE the export button ----
        st.markdown("### Value Proposition (Preview)")

        vp_cols = st.columns([2, 1, 1])
        with vp_cols[0]:
            st.caption("We’ll synthesize two versions: a SharePoint-grounded BCT value proposition with [n] footnotes, and a generic version without citations.")
        with vp_cols[1]:
            vp_k = st.slider("Evidence depth (Top-K from SharePoint)", min_value=8, max_value=40, value=int(ss.get("vp_k", 25)), step=1, key="vp_k")
        with vp_cols[2]:
            if st.button("Generate / Refresh Value Props", key="btn_vp"):
                with st.spinner("Synthesizing value propositions…"):
                    bct_vp_text, bct_vp_sources, generic_vp_text = _make_bct_value_props(cfg, oai, ss.get("rfp_facts") or {}, k=int(st.session_state["vp_k"]))
                ss.vp_bct_text = bct_vp_text
                ss.vp_bct_sources = bct_vp_sources
                ss.vp_generic_text = generic_vp_text

        # Show previews if available
        if ss.get("vp_bct_text") or ss.get("vp_generic_text"):
            st.markdown("#### BCT Value Proposition (SharePoint-grounded)")
            st.markdown(ss.get("vp_bct_text") or "_Not available_")
            if ss.get("vp_bct_sources"):
                st.caption("Sources: " + ", ".join(ss.get("vp_bct_sources", [])))

            st.markdown("#### Generic Value Proposition")
            st.markdown(ss.get("vp_generic_text") or "_Not available_")
        else:
            st.info("Click **Generate / Refresh Value Props** to preview the value propositions based on your SOW and SharePoint corpus.")

        # Export to Understanding DOCX
        c_dl1, c_dl2 = st.columns([1,1])
        with c_dl1:
            if st.button("Build Understanding DOCX (with Value Props)"):
                if not ss.get("rfp_facts"):
                    st.warning("No RFP facts available yet.")
                else:
                    with st.spinner("Synthesizing value propositions and compiling document…"):
                        bct_vp_text, bct_vp_sources, generic_vp_text = _make_bct_value_props(cfg, oai, ss.get("rfp_facts") or {}, k=25)
                        ud_bytes = build_understanding_docx(
                            facts=ss.get("rfp_facts") or {},
                            rfp_chat_messages=ss.get("rfp_chat_messages") or [],
                            sp_chat_messages=ss.get("sp_chat_messages") or [],
                            title="RFP Understanding Report",
                            add_toc=True,
                            # NEW: include the on-page previews
                            bct_vp_text=ss.get("vp_bct_text"),
                            bct_vp_sources=ss.get("vp_bct_sources"),
                            generic_vp_text=ss.get("vp_generic_text"),
                        )
                    st.session_state["understanding_docx_bytes"] = ud_bytes
                    st.success("Understanding document ready below.")

        with c_dl2:
            if st.session_state.get("understanding_docx_bytes"):
                st.download_button(
                    "⬇ Download Understanding DOCX",
                    data=st.session_state["understanding_docx_bytes"],
                    file_name="RFP_Understanding.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )


        # ---- Chat with your RFP (streaming, citations) ----
        st.divider()
        render_rfp_chat(cfg, oai)

        # -------- Section planning & quick-generate --------
        ctx_from_facts = facts_to_context(ss.rfp_facts or {})
        static_map_now = load_static_map(STATIC_MAP_FILE)
        plan = plan_dynamic_sections(ctx_from_facts, static_map_now, top_n=6)
        ss.section_plan = [p.to_dict() if hasattr(p, "to_dict") else vars(p) for p in plan]

        with st.expander("Show proposed dynamic plan", expanded=False):
            for p in plan:
                st.write(f"- **{p.label}** (priority {getattr(p, 'priority', 0):.2f}, tone {p.knobs.tone}, style {p.knobs.style})")

        st.markdown("#### Generate now from these facts")
        cgl1, cgl2 = st.columns([1,1])
        with cgl1:
            if st.button("Preload plan into Generation", key="preload_plan"):
                bp = plan_to_blueprint(plan, current_template_name=ss.get("gen_tpl"), tone=ctx_from_facts.rfp_type or "Professional")
                ss.pending_blueprint = bp; st.success("Dynamic plan queued. Switch to Proposal Generation tab if needed.")
        with cgl2:
            if st.button("Generate draft now (single click)", key="gen_now"):
                if ss.gen_tpl == "— choose a template —":
                    st.warning("Choose a template in the Proposal Generation tab first.")
                else:
                    available_static = [sec for sec in STATIC_SECTIONS if static_map_now.get(sec)]
                    dyn_sel = [p.label for p in plan]
                    final_order = available_static + [f"[Dyn] {x}" for x in dyn_sel]
                    template_path = str((TEMPLATE_DIR / ss.gen_tpl).resolve())
                    static_paths = {sec: str((STATIC_DIR / static_map_now[sec]).resolve()) for sec in available_static if static_map_now.get(sec)}
                    opts = BuildOptions(
                        use_anchors=True, template_has_headings=True, page_breaks=True,
                        include_sources=True, add_toc=True, rec_style="paragraphs",
                        top_k_default=6, top_k_per_section=25, tone="Professional",
                        static_heading_mode="demote", facts=ss.rfp_facts or {}
                    )
                    with st.spinner("Generating draft from current facts & template…"):
                        draft_bytes, recs_bytes, preview, meta = build_proposal(
                            template_path=template_path, static_paths=static_paths,
                            final_order=final_order, oai=oai, cfg=cfg, opts=opts
                        )
                    ss.out_draft_bytes = draft_bytes; ss.out_recs_bytes = recs_bytes; ss.dyn_recos_preview = preview
                    meta["template"] = ss.gen_tpl; meta["final_order"] = final_order; meta["timestamp"] = strftime("%Y-%m-%d %H:%M:%S", localtime())
                    ss.last_generation_meta = meta; st.success("Draft generated. See ‘Latest output’ in Proposal Generation tab.")

    else:
        st.info("Upload RFP documents above to enable facts, insights and chat.")

# ================= Proposal Generation =================
with tab_generation:
    if ss.get("pending_blueprint"):
        bp = ss.pending_blueprint or {}
        ss.gen_tpl = bp.get("template", ss.gen_tpl)
        if bp.get("static_sel") is not None: ss.gen_static_sel = bp.get("static_sel") or ss.gen_static_sel
        ss.gen_dyn_sel = bp.get("dyn_sel", ss.gen_dyn_sel)
        ss.gen_include_sources = bool(bp.get("include_sources", ss.gen_include_sources))
        ss.gen_top_k = int(bp.get("top_k", ss.gen_top_k))
        ss.gen_use_anchors = bool(bp.get("use_anchors", ss.gen_use_anchors))
        ss.gen_add_headings = bool(bp.get("add_headings", ss.gen_add_headings))
        ss.gen_rec_style = bp.get("rec_style", ss.gen_rec_style)
        ss.gen_per_section_k = int(bp.get("per_section_k", ss.gen_per_section_k))
        ss.gen_add_toc = bool(bp.get("add_toc", ss.gen_add_toc))
        ss.gen_page_breaks = bool(bp.get("page_breaks", ss.gen_page_breaks))
        ss.gen_tpl_has_headings = bool(bp.get("template_has_headings", ss.gen_tpl_has_headings))
        ss.pending_blueprint = None

    st.subheader("Proposal Generation")
    st.markdown('<div class="sticky"><span class="badge">Step 2</span> Single DOCX from Template + Static + Dynamic (SharePoint-grounded)</div>', unsafe_allow_html=True)

    with st.expander("SharePoint index status", expanded=False):
        chunks, vec_path = sp_index_stats(cfg); st.write(f"Index path: `{vec_path}` • Chunks: **{chunks}**")
        if chunks == 0: st.warning("SharePoint index is empty. Dynamic recommendations will be generic.")

    templates = sorted([f for f in os.listdir(TEMPLATE_DIR) if f.lower().endswith(".docx")])
    tpl_options = ["— choose a template —"] + templates
    try:
        tpl_index = tpl_options.index(ss.get("gen_tpl", "— choose a template —"))
    except ValueError:
        tpl_index = 0
    _ = st.selectbox("Template", options=tpl_options, index=tpl_index, key="gen_tpl",
                     help="Upload templates under Settings → Static Library → Proposal Templates.")

    static_map = load_static_map(STATIC_MAP_FILE)
    available_static = [sec for sec in STATIC_SECTIONS if static_map.get(sec)]

    if not ss.get("gen_static_sel"):
        ss.gen_static_sel = available_static

    st.markdown("#### Static sections (from your mapped .docx)")
    _ = st.multiselect("Select static sections", options=available_static, default=ss.get("gen_static_sel", available_static), key="gen_static_sel")

    st.markdown("#### Dynamic sections (SharePoint-grounded recommendations)")
    dynamic_pool = ["Executive Summary","Approach & Methodology","Staffing Plan & Roles","Transition & Knowledge Transfer","Service Levels & Governance",
                    "Assumptions & Exclusions","Risk & Mitigation Plan","Project Governance","Change Management","Quality Management","Compliance & Security","Timeline & Milestones"]
    blocked = set(ss.gen_static_sel); dynamic_pool = [x for x in dynamic_pool if x not in blocked]
    default_dyn = [x for x in ["Executive Summary", "Approach & Methodology", "Staffing Plan & Roles"] if x in dynamic_pool]
    _ = st.multiselect("Select dynamic sections", options=dynamic_pool, default=ss.get("gen_dyn_sel", default_dyn), key="gen_dyn_sel")

    st.markdown("#### Final order")
    combined_options = ss.gen_static_sel + [f"[Dyn] {x}" for x in ss.gen_dyn_sel]
    final_order_widget = st.multiselect("Pick the final sequence (drag to re-order)", options=combined_options, default=combined_options, key="gen_final_order_widget")
    final_order = final_order_widget

    _ = st.checkbox("Place at template anchors (SDT/Bookmark/Marker)", value=ss.get("gen_use_anchors", True), key="gen_use_anchors")
    _ = st.checkbox("Append 'Sources' line for dynamic sections", value=ss.get("gen_include_sources", True), key="gen_include_sources")
    _ = st.slider("Context Top-K (SharePoint)", min_value=3, max_value=12, value=int(ss.get("gen_top_k", 6)), step=1, key="gen_top_k")
    _ = st.selectbox("Recommendations style", options=["bullets","paragraphs"], index=["bullets","paragraphs"].index(ss.get("gen_rec_style","paragraphs")), key="gen_rec_style")
    _ = st.number_input("Per-section Top-K override (0 = use slider)", min_value=0, max_value=50, value=int(ss.get("gen_per_section_k", 25)), step=1, key="gen_per_section_k")
    _ = st.checkbox("Page break between sections", value=ss.get("gen_page_breaks", True), key="gen_page_breaks")
    _ = st.checkbox("Insert Table of Contents at top", value=ss.get("gen_add_toc", True), key="gen_add_toc")
    _ = st.checkbox("Template has headings at anchors (don’t add heading in dynamic content)", value=ss.get("gen_tpl_has_headings", True), key="gen_tpl_has_headings")

    st.markdown("#### Blueprints")
    bp_cols = st.columns([2,1,1,2])
    with bp_cols[0]:
        bp_name = st.text_input("Name a blueprint to save current setup", value="")
    with bp_cols[1]:
        if st.button("Save blueprint", use_container_width=True, disabled=(bp_name.strip()=="")):
            payload = {
                "template": ss.gen_tpl, "static_sel": ss.gen_static_sel, "dyn_sel": ss.gen_dyn_sel,
                "include_sources": bool(ss.gen_include_sources), "top_k": int(ss.gen_top_k), "tone": "Professional",
                "use_anchors": bool(ss.gen_use_anchors), "add_headings": bool(ss.gen_add_headings), "rec_style": ss.gen_rec_style,
                "per_section_k": int(ss.gen_per_section_k), "add_toc": bool(ss.gen_add_toc), "page_breaks": bool(ss.gen_page_breaks),
                "template_has_headings": bool(ss.gen_tpl_has_headings),
            }
            path = save_blueprint(BASE_DIR, bp_name.strip(), payload); st.success(f"Saved blueprint → {path}")
    with bp_cols[2]:
        choices = ["(choose)"] + list_blueprints(BASE_DIR); chosen = st.selectbox("Load blueprint", options=choices, index=0)
    with bp_cols[3]:
        if st.button("Apply blueprint", use_container_width=True, disabled=(chosen=="(choose)")):
            bp = load_blueprint(BASE_DIR, chosen)
            if bp: ss.pending_blueprint = bp; st_rerun_compat()

    disabled = (ss.gen_tpl == "— choose a template —") or (len(final_order) == 0)

    if st.button("Generate Draft + Recommendations", disabled=disabled, use_container_width=True):
        if ss.gen_tpl == "— choose a template —":
            st.warning("Please choose a template.")
        elif len(final_order) == 0:
            st.warning("Choose at least one section in the final order.")
        else:
            ss.dyn_recos_preview = {}
            template_path = str((TEMPLATE_DIR / ss.gen_tpl).resolve())
            static_map_now = load_static_map(STATIC_MAP_FILE)
            static_paths = {sec: str((STATIC_DIR / static_map_now[sec]).resolve()) for sec in ss.gen_static_sel if static_map_now.get(sec)}
            opts = BuildOptions(
                use_anchors=ss.gen_use_anchors, template_has_headings=ss.gen_tpl_has_headings, page_breaks=ss.gen_page_breaks,
                include_sources=ss.gen_include_sources, add_toc=ss.gen_add_toc, rec_style=ss.gen_rec_style,
                top_k_default=int(ss.gen_top_k), top_k_per_section=int(ss.gen_per_section_k), tone="Professional",
                static_heading_mode="demote", facts=ss.rfp_facts or {}
            )
            with st.spinner("Composing draft and recommendations…"):
                draft_bytes, recs_bytes, preview, meta = build_proposal(
                    template_path=template_path, static_paths=static_paths,
                    final_order=final_order, oai=oai, cfg=cfg, opts=opts
                )
            ss.out_draft_bytes = draft_bytes; ss.out_recs_bytes = recs_bytes; ss.dyn_recos_preview = preview
            meta["template"] = ss.gen_tpl; meta["final_order"] = final_order; meta["timestamp"] = strftime("%Y-%m-%d %H:%M:%S", localtime())
            ss.last_generation_meta = meta; st.success("Draft generated. See the 'Latest output' panel below to download anytime.")

    st.markdown("### Latest output")
    with st.container(border=True):
        meta = ss.last_generation_meta or {}
        if ss.out_draft_bytes:
            tpl_name = meta.get("template", "—"); ts = meta.get("timestamp", "—"); order_preview = ", ".join(meta.get("final_order", [])) or "—"
            st.caption(f"Generated: **{ts}** • Template: **{tpl_name}**")
            with st.expander("Show final sequence used", expanded=False): st.write(order_preview)
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "⬇ Download Draft DOCX",
                    data=ss.out_draft_bytes,
                    file_name="Draft_Response_Static+Dynamic.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
            with c2:
                if ss.out_recs_bytes:
                    st.download_button(
                        "⬇ Download Recommendations DOCX",
                        data=ss.out_recs_bytes,
                        file_name="Dynamic_Recommendations.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                    )

            if ss.dyn_recos_preview:
                st.markdown("#### Dynamic recommendations used (preview)")
                all_md = ["# Dynamic Recommendations (Preview)\n"]
                for sec_name, pkg in ss.dyn_recos_preview.items():
                    st.markdown(f"**{sec_name}**")
                    st.markdown(pkg["md"])
                    if pkg.get("evidence"):
                        with st.expander(f"Show retrieval diagnostics — {sec_name}", expanded=False):
                            for i, ev in enumerate(pkg["evidence"], start=1):
                                st.markdown(f"**[{i}]** `{_fmt_src(ev)}`{_fmt_why(ev)}")
                                st.code(_fmt_text(ev)[:1200], language="text")
                    if pkg.get("sources"):
                        st.caption("Sources: " + ", ".join(pkg["sources"]))
                    all_md.append(f"## {sec_name}\n\n{pkg['md']}\n")
                    if pkg.get("sources"):
                        all_md.append(f"_Sources: {', '.join(pkg['sources'])}_\n")
                md_blob = "\n".join(all_md).encode("utf-8")
                st.download_button("⬇ Download recommendations (Markdown)", data=md_blob, file_name="dynamic_recommendations.md", mime="text/markdown")
        else:
            st.caption("No generated files yet. Click **Generate** above to create your draft.")

# ================= Settings =================
with tab_settings:
    st.subheader("Settings")
    with st.expander("Static Library (manage section .docx files)", expanded=False):
        STATIC_DIR.mkdir(parents=True, exist_ok=True); TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
        static_map_now = load_static_map(STATIC_MAP_FILE)

        up_cols = st.columns([2,2,1])
        with up_cols[0]:
            static_uploads = st.file_uploader("Upload .docx files for your static library", type=["docx"], accept_multiple_files=True)
        with up_cols[1]:
            if st.button("Save uploads"):
                if static_uploads:
                    saved = []
                    for f in static_uploads:
                        out = STATIC_DIR / f.name; out.write_bytes(f.getvalue()); saved.append(f.name)
                    st.success(f"Uploaded: {', '.join(saved)}")
                else:
                    st.info("No files to save.")
        with up_cols[2]:
            st.caption(f"Folder: `{STATIC_DIR.resolve()}`")

        st.markdown("##### Library Files")
        lib_files = list_files(STATIC_DIR, exts=("docx",))
        if lib_files:
            st.write("\n".join([f"- {x}" for x in lib_files]))
        else:
            st.info("No static files yet. Upload .docx above.")

        st.markdown("---"); st.markdown("##### Map Sections → File")
        for sec in STATIC_SECTIONS:
            cols = st.columns([2,2,2,1])
            with cols[0]:
                st.write(f"**{sec}**")
            with cols[1]:
                current = static_map_now.get(sec, "")
                choice = st.selectbox(
                    "File", options=["(none)"] + lib_files,
                    index=(["(none)"] + lib_files).index(current) if current in lib_files else 0,
                    key=f"map_{sec}"
                )
            with cols[2]:
                if st.button("Preview", key=f"prev_{sec}"):
                    fn = st.session_state.get(f"map_{sec}", "(none)")
                    if fn != "(none)":
                        text = extract_text(str(STATIC_DIR / fn))[:1200]
                        st.info(f"Preview of **{fn}** (first ~1200 chars)\n\n{text}")
                    else:
                        st.info("No file mapped.")
            with cols[3]:
                pass

        if st.button("Save Mapping"):
            for sec in STATIC_SECTIONS:
                val = st.session_state.get(f"map_{sec}", "(none)")
                static_map_now[sec] = "" if val == "(none)" else val
            save_static_map(STATIC_MAP_FILE, static_map_now)
            st.success("Static section mapping saved.")

        st.markdown("---"); st.markdown("##### Proposal Templates (.docx)")
        tpl_upload = st.file_uploader("Upload a proposal template (.docx)", type=["docx"], accept_multiple_files=False)
        if tpl_upload and st.button("Save Template"):
            out = TEMPLATE_DIR / tpl_upload.name
            out.write_bytes(tpl_upload.getvalue())
            st.success(f"Saved template: {tpl_upload.name}")
        st.caption(f"Templates folder: `{TEMPLATE_DIR.resolve()}`")

    with st.expander("SharePoint (sites, ingestion & maintenance)", expanded=False):
        ss.sp_urls = ss.get("sp_urls", load_saved_urls(SP_URL_STORE))
        urls = ss.sp_urls
        if not urls:
            st.info("No SharePoint site URLs saved yet. Add one below to get started.")
        if urls:
            labels = [canonicalize_site_url(u) for u in urls]
            ss.sp_selected_idx = st.selectbox(
                "Saved SharePoint Sites",
                options=list(range(len(urls))),
                format_func=lambda i: labels[i],
                index=min(ss.sp_selected_idx, len(urls) - 1)
            )
            selected_site_raw = urls[min(ss.sp_selected_idx, len(urls)-1)]
            selected_site_key = canonicalize_site_url(selected_site_raw)
            files_for_url = ss.sp_ingested_map.get(selected_site_key, [])
            st.markdown("###### Files Ingested for Selected Site")
            if files_for_url:
                q = st.text_input("Filter files (type to search)", value="")
                flist = [f for f in files_for_url if (q.lower() in os.path.basename(f).lower())]
                st.write(f"{len(flist)} file(s) shown")
                cols = st.columns(3)
                for i, f in enumerate(flist):
                    with cols[i % 3]:
                        st.markdown(f"- `{os.path.basename(f)}`")
                if st.checkbox("Show full paths", value=False, key="show_full_sp_paths"):
                    for f in flist:
                        st.code(f, language="text")
            else:
                st.info("No files recorded yet for this site. Use Pull & Index below.")

        st.markdown("###### Add or Edit SharePoint Site")
        edit_mode = st.checkbox("Edit selected?", value=False, disabled=not urls)
        current_url = urls[ss.sp_selected_idx] if (edit_mode and urls) else ""
        new_url = st.text_input(
            "SharePoint Site URL (accepts full .aspx paths or /sites/<name>)",
            value=current_url,
            placeholder="https://contoso.sharepoint.com/sites/YourSite or .../SitePages/Home.aspx"
        )
        d1, d2, d3 = st.columns([1, 1, 1])
        with d1:
            if st.button("Save URL"):
                if not new_url.strip():
                    st.warning("Please enter a URL.")
                else:
                    can = canonicalize_site_url(new_url.strip())
                    urls = upsert_url(urls, can, ss.sp_selected_idx if edit_mode and urls else None)
                    ss.sp_urls = urls
                    save_saved_urls(SP_URL_STORE, urls)
                    st.success("URL saved.")
        with d2:
            if st.button("Add as New"):
                if not new_url.strip():
                    st.warning("Please enter a URL.")
                else:
                    can = canonicalize_site_url(new_url.strip())
                    urls = upsert_url(urls, can, None)
                    ss.sp_urls = urls
                    save_saved_urls(SP_URL_STORE, urls)
                    st.success("URL added.")
        with d3:
            if st.button("Delete Selected", disabled=not urls):
                urls = delete_url(urls, ss.sp_selected_idx)
                ss.sp_urls = urls
                ss.sp_selected_idx = 0
                save_saved_urls(SP_URL_STORE, urls)
                st.success("URL deleted.")

        st.markdown("---"); st.markdown("###### Pull & Index")
        exts = st.multiselect("File types to include (download filter)", ["pdf", "docx", "pptx", "xlsx", "xls", "txt"], default=["pdf", "docx", "pptx"])
        chunk_size = st.number_input("Chunk size for vectorization", min_value=300, max_value=4000, value=getattr(cfg, "CHUNK_SIZE", 1000), step=100)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            do_pull = st.button("Pull & Index from Selected URL")
        with c2:
            do_refresh = st.button("Re-index (Clear SP collection then Pull)")
        with c3:
            do_clear = st.button("Clear SP Vector Store")

        selected_site = urls[min(ss.sp_selected_idx, len(urls)-1)] if urls else None
        if do_clear:
            wipe_sp_store(cfg)
            st.success("SharePoint vector store cleared.")
        if do_pull or do_refresh:
            if not selected_site:
                st.warning("Please select or add a SharePoint site first.")
            else:
                cfg_dict = vars(cfg).copy()
                cfg_dict["SP_SITE_URL"] = canonicalize_site_url(selected_site)
                cfg_dict["CHUNK_SIZE"] = int(chunk_size)
                dynamic_cfg = Config(**cfg_dict)
                if do_refresh:
                    wipe_sp_store(dynamic_cfg)
                with st.spinner("Contacting SharePoint, fetching files, and indexing…"):
                    files = ingest_sharepoint(dynamic_cfg, include_exts=exts)
                if files:
                    ss.sp_ingested_files = files
                    ss.temp_files.extend(files)
                    key = canonicalize_site_url(selected_site)
                    ss.sp_ingested_map[key] = files
                    save_sp_ingested_map(SP_INGEST_MAP_FILE, ss.sp_ingested_map)
                    _ = init_sharepoint_store(dynamic_cfg)
                    st.success(f"Ingested and indexed {len(files)} files from SharePoint.")
                    if st.checkbox("Show ingested file names", value=False, key="show_ingested_names"):
                        for f in files:
                            st.write(os.path.basename(f))
                else:
                    st.info("No files were found on this site with the selected filters.")

    with st.expander("OCR Controls & Diagnostics", expanded=False):
        ss.ocr_enable = st.checkbox("Enable OCR fallback for PDFs (uses Tesseract if available)", value=bool(ss.ocr_enable))
        st.caption("When enabled, if a PDF yields no text, we'll attempt image OCR as a fallback.")
