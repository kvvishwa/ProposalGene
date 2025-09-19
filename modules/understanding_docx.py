# modules/understanding_docx.py
from __future__ import annotations

from io import BytesIO
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import re

from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

# -----------------------------------------------------------------------------
# Safety: strip XML-illegal control chars that can break python-docx
# -----------------------------------------------------------------------------
_XML_ILLEGAL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

# -----------------------------------------------------------------------------
# String utilities (robust against ints, lists, dicts)
# -----------------------------------------------------------------------------

def _t(v: Any) -> str:
    """Coerce any value to a safe, printable unicode string."""
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        # Join lists into a human-friendly line
        return ", ".join(_t(x) for x in v)
    if isinstance(v, dict):
        # Keep it readable if something bubbles up unexpectedly
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    return str(v)


def _clean(s: Any) -> str:
    """Normalize whitespace after coercion to string."""
    s = _t(s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


def _san(s: Optional[str]) -> str:
    s = _t(s)
    if not s:
        return ""
    return _XML_ILLEGAL.sub("", s)

# -----------------------------------------------------------------------------
# Word helpers
# -----------------------------------------------------------------------------

def _add_toc(doc: Document, title: str = "Table of Contents") -> None:
    doc.add_heading(title, level=1)
    p = doc.add_paragraph()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), 'TOC \\o "1-3" \\h \\z \\u')
    p._p.append(fld)


def _add_kv(doc: Document, k: str, v: Any) -> None:
    if not (k or v):
        return
    p = doc.add_paragraph()
    r1 = p.add_run(_san(k or ""))
    r1.bold = True
    val = _san(_t(v))
    if val:
        p.add_run(": ")
        p.add_run(val)


def _add_list(doc: Document, items: List[Any], bullet: bool = True) -> None:
    for it in items or []:
        txt = _san(_t(it))
        if not txt:
            continue
        p = doc.add_paragraph(txt)
        try:
            p.style = "List Bullet" if bullet else "List Number"
        except Exception:
            pass


def _add_mdish_block(doc: Document, text: str) -> None:
    """
    Very light Markdown-to-Word:
      - lines starting with -, *, • → bullet
      - blank lines preserved
    """
    for raw in (_san(text).splitlines() if text else []):
        ln = raw.strip()
        if not ln:
            doc.add_paragraph("")
            continue
        if ln.startswith(("- ", "* ", "• ")):
            p = doc.add_paragraph(ln[2:].strip())
            try:
                p.style = "List Bullet"
            except Exception:
                pass
        else:
            doc.add_paragraph(ln)


def _format_date(d: Any) -> str:
    """Support dict {{date,time,tz}} OR plain string."""
    if isinstance(d, dict):
        date = _t(d.get("date") or d.get("datetime") or d.get("due") or "")
        time = _t(d.get("time"))
        tz = _t(d.get("tz") or d.get("timezone"))
        return " ".join(x for x in [date, time, tz] if x)
    return _san(_t(d or ""))

# -----------------------------------------------------------------------------
# Section renderers
# -----------------------------------------------------------------------------

def _render_scope(doc: Document, facts: Dict[str, Any]) -> None:
    """Add a dedicated 'Scope' section with in/out lists or a raw fallback."""
    scope = (facts or {}).get("scope") or {}
    if not scope:
        return
    doc.add_heading("7. Scope", level=1)
    if scope.get("in_scope") or scope.get("out_of_scope"):
        if scope.get("in_scope"):
            doc.add_paragraph("In Scope")
            _add_list(doc, scope.get("in_scope") or [])
        if scope.get("out_of_scope"):
            doc.add_paragraph("Out of Scope")
            _add_list(doc, scope.get("out_of_scope") or [])
    elif scope.get("scope_raw"):
        doc.add_paragraph("Scope (Detected)")
        _add_list(doc, scope.get("scope_raw") or [])


def _render_facts(doc: Document, facts: Dict[str, Any]) -> None:
    if not isinstance(facts, dict):
        facts = {}

    # 1) Solicitation
    doc.add_heading("1. Solicitation", level=1)
    sol = facts.get("solicitation", {}) or {}
    _add_kv(doc, "Issuing Entity", sol.get("issuing_entity_name", ""))
    _add_kv(doc, "Entity Type", sol.get("issuing_entity_type", ""))
    _add_kv(doc, "Department/Office", sol.get("department_or_office", ""))
    _add_kv(doc, "Solicitation ID", sol.get("solicitation_id", ""))
    _add_kv(doc, "Title", sol.get("solicitation_title", ""))
    _add_kv(doc, "Communication Policy", sol.get("communication_policy", ""))

    # 2) Points of Contact
    doc.add_heading("2. Points of Contact", level=1)
    for i, poc in enumerate(facts.get("points_of_contact", []) or [], start=1):
        doc.add_heading(f"POC {i}", level=2)
        _add_kv(doc, "Primary", "Yes" if poc.get("primary") else "No")
        _add_kv(doc, "Name", poc.get("name", ""))
        _add_kv(doc, "Title", poc.get("title", ""))
        _add_kv(doc, "Email", poc.get("email", ""))
        _add_kv(doc, "Phone", poc.get("phone", ""))
        _add_kv(doc, "Address", poc.get("address", ""))
        _add_kv(doc, "Notes", poc.get("notes", ""))

    # 3) Schedule
    doc.add_heading("3. Schedule & Deadlines", level=1)
    sch = facts.get("schedule", {}) or {}
    _add_kv(doc, "Release Date", _format_date(sch.get("release_date")))
    pre = sch.get("pre_bid_conference", {}) or {}
    _add_kv(doc, "Pre-bid Conference", f"{_format_date(pre.get('datetime',''))} @ {_san(pre.get('location',''))}")
    site = sch.get("site_visit", {}) or {}
    _add_kv(doc, "Site Visit", f"{_format_date(site.get('datetime',''))} @ {_san(site.get('location',''))}")
    _add_kv(doc, "Q&A Deadline", _format_date(sch.get("qna_deadline")))
    _add_kv(doc, "Addendum Final Date", _format_date(sch.get("addendum_final_date")))
    _add_kv(doc, "Submission Due", _format_date(sch.get("submission_due")))
    _add_kv(doc, "Award Target Date", _format_date(sch.get("award_target_date")))
    _add_kv(doc, "Anticipated Start Date", _format_date(sch.get("anticipated_start_date")))
    _add_kv(doc, "Proposal Validity (days)", _t(sch.get("proposal_validity_days") or ""))
    _add_kv(doc, "Contract Term", sch.get("contract_term", ""))
    _add_kv(doc, "Renewal Options", sch.get("renewal_options", ""))

    # 4) Submission Instructions
    doc.add_heading("4. Submission Instructions", level=1)
    sub = facts.get("submission_instructions", {}) or {}
    _add_kv(doc, "Method", sub.get("method", ""))
    _add_kv(doc, "Portal", f"{_san(sub.get('portal_name',''))} {_san(sub.get('portal_url',''))}".strip())
    _add_kv(doc, "Email", sub.get("email_address", ""))
    _add_kv(doc, "Physical Address", sub.get("physical_address", ""))
    _add_kv(doc, "Copies Required", sub.get("copies_required", ""))
    _add_list(doc, sub.get("format_requirements") or [])
    _add_kv(doc, "Labeling Instructions", sub.get("labeling_instructions", ""))
    _add_kv(doc, "Registration Requirements", sub.get("registration_requirements", ""))
    _add_kv(doc, "Late Submissions Policy", sub.get("late_submissions_policy", ""))

    # 5) Minimum Qualifications
    doc.add_heading("5. Minimum Qualifications", level=1)
    mq = facts.get("minimum_qualifications", {}) or {}
    _add_list(doc, mq.get("licenses_certifications") or [])
    _add_kv(doc, "Years Experience", mq.get("years_experience", ""))
    _add_kv(doc, "Similar Projects", mq.get("similar_projects", ""))
    _add_list(doc, mq.get("insurance_requirements") or [])
    _add_list(doc, mq.get("bonding") or [])
    _add_kv(doc, "Financials", mq.get("financials", ""))
    _add_list(doc, mq.get("other_mandatory") or [])

    # 6) Proposal Organization
    doc.add_heading("6. Proposal Organization", level=1)
    org = facts.get("proposal_organization", {}) or {}
    _add_list(doc, org.get("required_sections") or [])
    for pl in (org.get("page_limits") or []):
        _add_kv(doc, f"Page Limit — {pl.get('section','')}", pl.get("limit",""))
    _add_list(doc, org.get("mandatory_forms") or [])
    _add_kv(doc, "Pricing Format", org.get("pricing_format", ""))
    _add_kv(doc, "Exceptions Policy", org.get("exceptions_policy", ""))

    # 7) Scope
    _render_scope(doc, facts)

    # 8) Evaluation & Selection
    doc.add_heading("8. Evaluation & Selection", level=1)
    ev = facts.get("evaluation_and_selection", {}) or {}
    for c in ev.get("criteria") or []:
        nm = c.get("name", "")
        wt = c.get("weight", "")
        desc = c.get("description", "")
        _add_kv(doc, f"Criterion — {nm}", f"{_t(wt)} {desc}".strip())
    _add_list(doc, ev.get("pass_fail_criteria") or [])
    oral = ev.get("oral_presentations", {}) or {}
    demo = ev.get("demonstrations", {}) or {}
    _add_kv(doc, "Oral Presentations", f"{'required' if oral.get('required') else 'optional'} — {oral.get('notes','')}".strip())
    _add_kv(doc, "Demonstrations", f"{'required' if demo.get('required') else 'optional'} — {demo.get('notes','')}".strip())
    _add_kv(doc, "BAFO", f"{'allowed' if (ev.get('best_and_final_offers') or {}).get('allowed') else 'not indicated'}")
    _add_kv(doc, "Negotiations", f"{'allowed' if (ev.get('negotiations') or {}).get('allowed') else 'not indicated'}")
    _add_kv(doc, "Selection Summary", ev.get("selection_process_summary", ""))
    _add_kv(doc, "Protest Procedure", ev.get("protest_procedure", ""))

    # 9) Contract & Compliance
    doc.add_heading("9. Contract & Compliance", level=1)
    cc = facts.get("contract_and_compliance", {}) or {}
    _add_list(doc, cc.get("key_terms") or [])
    _add_list(doc, cc.get("security_privacy") or [])
    if cc.get("sow_summary"):
        doc.add_paragraph(_san(cc.get("sow_summary")))

    # 10) Missing / Notes
    miss = facts.get("missing_notes") or []
    if miss:
        doc.add_heading("10. Missing / Notes", level=1)
        _add_list(doc, miss)


def _render_value_prop(
    doc: Document,
    bct_vp_text: Optional[str],
    bct_vp_sources: Optional[List[str]],
    generic_vp_text: Optional[str],
) -> None:
    # 11) BCT Value Proposition (SP-grounded)
    if bct_vp_text:
        doc.add_heading("11. Value Proposition — BCT (SharePoint-grounded)", level=1)
        _add_mdish_block(doc, bct_vp_text)
        if bct_vp_sources:
            doc.add_paragraph("Sources: " + ", ".join(_san(s) for s in bct_vp_sources))

    # 12) Generic Value Proposition
    if generic_vp_text:
        doc.add_heading("12. Value Proposition — Generic", level=1)
        _add_mdish_block(doc, generic_vp_text)


def _render_chat(doc: Document, title: str, messages: List[Dict[str, Any]]) -> None:
    doc.add_heading(title, level=1)
    if not messages:
        doc.add_paragraph("— no messages —")
        return
    for turn in messages:
        role = _t(turn.get("role")).strip().lower()
        content = _san(turn.get("content") or "")
        p = doc.add_paragraph()
        r = p.add_run(("User: " if role == "user" else "Assistant: "))
        r.bold = True
        _add_mdish_block(doc, content)
        srcs = turn.get("sources") or []
        if srcs:
            doc.add_paragraph("Sources: " + ", ".join(_san(s) for s in srcs))

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def build_understanding_docx(
    facts: Dict[str, Any],
    rfp_chat_messages: List[Dict[str, Any]],
    sp_chat_messages: List[Dict[str, Any]],
    title: str = "RFP Understanding Report",
    add_toc: bool = True,
    *,
    bct_vp_text: Optional[str] = None,
    bct_vp_sources: Optional[List[str]] = None,
    generic_vp_text: Optional[str] = None,
) -> bytes:
    """
    Returns a DOCX as bytes containing:
      - Title page
      - (optional) Table of Contents
      - RFP Facts Summary (with Scope)
      - Value Proposition (BCT grounded + generic)
      - RFP Chat transcript
      - SharePoint Chat transcript
    """
    doc = Document()

    # Title page
    t = doc.add_paragraph(title)
    try:
        t.style = "Title"
    except Exception:
        pass
    ts = doc.add_paragraph(datetime.now().strftime("%Y-%m-%d %H:%M"))
    ts.alignment = WD_ALIGN_PARAGRAPH.LEFT

    # ToC
    if add_toc:
        _add_toc(doc)

    # Facts (includes Scope)
    _render_facts(doc, facts or {})

    # Value prop sections (after facts)
    _render_value_prop(doc, bct_vp_text, bct_vp_sources, generic_vp_text)

    # Chats
    _render_chat(doc, "RFP Chat Transcript", rfp_chat_messages or [])
    _render_chat(doc, "SharePoint Chat Transcript", sp_chat_messages or [])

    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()

# ----------------------------- Dynamic Sections Recommendation -----------------------------
import json
import pandas as pd
import streamlit as st
from openai import OpenAI

def _recommend_dynamic_sections(cfg, oai: OpenAI):
    st.markdown("### 🧩 Recommend Dynamic Sections (with weights)")
    st.caption("SharePoint-grounded where possible; weights 1–5 indicate importance. Click 'Apply' to send to Generation.")

    if st.button("Recommend Sections"):
        facts = st.session_state.get("rfp_facts") or {}
        # Keep prompt short; the uploaded RFP and SP evidence have already shaped facts + chats
        prompt = (
            "Return JSON with key 'sections' as a list of objects {section, weight, rationale}. "
            "Weights are integers 1-5 (importance for this RFP). Avoid sections already covered as static.\n\n"
            f"FACTS:\n{json.dumps(facts)[:4000]}\n\n"
            "Typical sections: Executive Summary; Understanding of Requirements; Approach & Methodology; "
            "Transition / Mobilization; Risk Management; Security & Compliance; Staffing & Key Personnel; "
            "Past Performance; Commercials & Assumptions."
        )
        try:
            res = oai.chat.completions.create(
                model=getattr(cfg, "ANALYSIS_MODEL", "gpt-4o"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=900,
            )
            raw = (res.choices[0].message.content or "").strip()
            # tolerate fenced / extra text
            import re, json as _json
            m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, flags=re.I)
            if m: raw = m.group(1)
            data = _json.loads(raw)
            rows = data.get("sections") or []
        except Exception as e:
            rows = []
            st.error(f"Section recommendation failed: {e}")

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            # keep a preview for transparency
            st.session_state["dyn_recos_preview"] = rows

            # PUSH TO GENERATION
            picked = [r.get("section", "") for r in rows if (r.get("weight") or 0) >= 3]
            weights = {r.get("section", ""): int(r.get("weight") or 0) for r in rows if r.get("section")}
            st.session_state["gen_dyn_sel"] = picked
            st.session_state["gen_dyn_weights"] = weights
            st.success("Dynamic sections applied for Generation (>=3 weight).")


