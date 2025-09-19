from __future__ import annotations
import json, re
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
from openai import OpenAI
import json as jsonlib

from modules.app_helpers import (
    save_rfp_facts_json,
    rag_answer_uploaded,
    get_passages_for_query,
)
from modules.ui_helpers import render_rfp_facts
from modules.vectorstore import init_uploaded_store
from modules.understanding_docx import build_understanding_docx

SCHEMA_HINT = """
Return strict JSON with keys:
- solicitation: {issuing_entity_name, issuing_entity_type, department_or_office, solicitation_id, solicitation_title, communication_policy}
- points_of_contact: [ {primary, name, title, email, phone, address, notes} ]
- schedule: {release_date, pre_bid_conference:{datetime,location}, site_visit:{datetime,location}, qna_deadline, addendum_final_date, submission_due, award_target_date, anticipated_start_date, proposal_validity_days, contract_term, renewal_options}
- submission_instructions: {method, portal_name, portal_url, email_address, physical_address, copies_required, format_requirements:[...], labeling_instructions, registration_requirements, late_submissions_policy}
- minimum_qualifications: {licenses_certifications:[...], years_experience, similar_projects, insurance_requirements:[...], bonding:[...], financials, other_mandatory:[...]}
- proposal_organization: {required_sections:[...], page_limits:[{section,limit}], mandatory_forms:[...], pricing_format, exceptions_policy}
- scope: {in_scope:[...], out_of_scope:[...], scope_raw:[...]}  # any of these
- evaluation_and_selection: {criteria:[{name,weight,description}], pass_fail_criteria:[...], oral_presentations:{required,notes}, demonstrations:{required,notes}, best_and_final_offers:{allowed}, negotiations:{allowed}, selection_process_summary, protest_procedure}
- contract_and_compliance: {key_terms:[...], security_privacy:[...], sow_summary}
- missing_notes:[...]
""".strip()


def _safe_dumps(obj, limit=4000):
    try:
        s = jsonlib.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    return s[:limit]

# ----------------------------- JSON helpers -----------------------------
def _coerce_json(text: str):
    s = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s, flags=re.I)
    if m: s = m.group(1)
    start = s.find("{")
    if start != -1:
        depth = 0; end = None
        for i, ch in enumerate(s[start:], start):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1; break
        if end: s = s[start:end]
    obj = json.loads(s)
    if isinstance(obj, str):
        try: obj = json.loads(obj)
        except Exception: pass
    return obj

def _ensure_dict(obj):
    return obj if isinstance(obj, dict) else {"_raw": obj}

# ----------------------------- Chat (RFP) -----------------------------
_DEF_TOPK = 6

def _render_rfp_chat(cfg, oai: OpenAI):
    st.markdown("### 💬 Ask Anything — Chat with your RFP")
    st.caption("Grounded on the uploaded RFP docs in this session.")
    store = st.session_state.get("up_store")
    if not store:
        st.info("Upload and vectorize an RFP first.")
        return
    msgs = st.session_state.setdefault("rfp_chat_messages", [])
    q = st.text_input("Your question", key="rfp_q", placeholder="e.g., What are the submission requirements and page limits?")
    top_k = st.number_input("Top-K", min_value=3, max_value=20, value=_DEF_TOPK, step=1)
    if st.button("Ask (RFP)", type="primary") and q.strip():
        prompt = f"Context focus cues: {q}\n\nAnswer the question briefly using only context."
        try:
            ans, srcs = rag_answer_uploaded(store, oai, cfg, prompt, top_k=int(top_k))
        except Exception as e:
            ans, srcs = (f"(LLM/RAG error) {e}", [])
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": ans, "sources": srcs})
    for m in msgs:
        with st.chat_message(m.get("role", "assistant")):
            st.markdown(m.get("content", ""))
            srcs = m.get("sources") or []
            if srcs:
                st.caption("Sources: " + ", ".join(srcs))

# ----------------------------- Chat (Open Web style) -----------------------------
def _render_web_chat(cfg, oai: OpenAI):
    st.markdown("### 🌐 Research Chat — Open Internet (model-only)")
    st.caption("Uses the model's general knowledge. (No live browsing wired here.)")
    msgs = st.session_state.setdefault("web_chat_messages", [])
    q = st.text_input("Your question (web)", key="web_q", placeholder="e.g., What are current best practices for ERP rollouts in public sector?")
    if st.button("Ask (Web)") and q.strip():
        sys = "You are a precise research assistant. Cite likely sources in-text when known."
        try:
            res = oai.chat.completions.create(
                model=getattr(cfg, "ANALYSIS_MODEL", "gpt-4o"),
                messages=[{"role":"system","content":sys},{"role":"user","content":q}],
                temperature=0.2,
                max_tokens=min(1100, getattr(cfg, "MAX_TOKENS", 2200)),
            )
            ans = (res.choices[0].message.content or "").strip()
        except Exception as e:
            ans = f"(LLM error) {e}"
        msgs.append({"role":"user","content":q}); msgs.append({"role":"assistant","content":ans})
    for m in msgs:
        with st.chat_message(m.get("role","assistant")):
            st.markdown(m.get("content",""))

# ----------------------------- Value Props -----------------------------
def _generate_vp_from_sp(cfg, oai: OpenAI):
    st.markdown("#### 🔎 Value Proposition from SharePoint (evidence-grounded)")
    st.caption("Synthesizes a crisp BCT value prop from your indexed SharePoint evidence.")
    cues = st.text_input("Focus (comma-separated)", value="experience, accelerators, methodology, success metrics")
    if st.button("Generate VP (SharePoint)"):
        try:
            parts = get_passages_for_query(cfg, cues, k=12, store="sharepoint")
            ctx = "\n\n---\n\n".join(p.get("text","")[:1600] for p in parts)
            sys = "Write a concise customer-facing value proposition for BCT in 6-8 bullets using ONLY the context."
            res = oai.chat.completions.create(
                model=getattr(cfg, "ANALYSIS_MODEL","gpt-4o"),
                messages=[{"role":"system","content":sys},{"role":"user","content":ctx}],
                temperature=0.2,
                max_tokens=600,
            )
            vp = (res.choices[0].message.content or "").strip()
            st.session_state.setdefault("rfp_facts", {})["bct_vp_text"] = vp
            st.session_state["rfp_facts"]["bct_vp_sources"] = [p.get("source","") for p in parts if p.get("source")]
            st.success("BCT value prop generated.")
            st.markdown(vp)
        except Exception as e:
            st.error(f"VP generation failed: {e}")

def _generate_vp_from_web(cfg, oai: OpenAI):
    st.markdown("#### 🌐 Value Proposition from Open Internet (model knowledge)")
    st.caption("No live browsing here; uses the model's general knowledge to suggest positioning.")
    industry = st.text_input("Target industry/domain (optional)", value="Public Sector / Government")
    if st.button("Generate VP (Open Internet)"):
        facts = st.session_state.get("rfp_facts") or {}
        title = ((facts.get("solicitation") or {}).get("solicitation_title") or "the opportunity")
        prompt = (
            f"Draft a persuasive value proposition (6-8 bullets) tailored for {industry} responding to '{title}'. "
            "Emphasize measurable outcomes, risk mitigation, compliance, accelerators, and case proof."
        )
        try:
            res = oai.chat.completions.create(
                model=getattr(cfg, "ANALYSIS_MODEL","gpt-4o"),
                messages=[{"role":"user","content":prompt}],
                temperature=0.4,
                max_tokens=600,
            )
            vp = (res.choices[0].message.content or "").strip()
            st.session_state.setdefault("rfp_facts", {})["generic_vp_text"] = vp
            st.success("Open-internet style value prop generated.")
            st.markdown(vp)
        except Exception as e:
            st.error(f"Open-internet VP failed: {e}")

# ----------------------------- Dynamic Sections Recommendation -----------------------------
def _recommend_dynamic_sections(cfg, oai: OpenAI):
    st.markdown("### 🧩 Recommend Dynamic Sections (with weights)")
    st.caption("Suggests which dynamic sections to include and assigns weights. Click 'Apply' to send to Generation module.")
    if st.button("Recommend Sections"):
        facts =st.session_state.get("rfp_facts") or {}
        prompt = (
            "Return JSON list under key 'sections' where each item is {section, weight, rationale}. "
            "Weights are integers 1-5 (importance for this RFP). Avoid sections already mapped as static.\n\n"
            f"FACTS:\n{_safe_dumps(facts)}\n\n"
            "Examples of sections: Executive Summary, Understanding of Requirements, Approach & Methodology, Transition Plan, Risk Management, Security & Compliance, Staffing & Key Personnel, Past Performance, Commercials & Assumptions."
        )
        try:
            res = oai.chat.completions.create(
                model=getattr(cfg, "ANALYSIS_MODEL","gpt-4o"),
                messages=[{"role":"user","content":prompt}],
                temperature=0.2,
                max_tokens=900,
            )
            raw = (res.choices[0].message.content or "").strip()
            data = _coerce_json(raw)
            rows = data.get("sections") or []
        except Exception as e:
            rows = []
            st.error(f"Section recommendation failed: {e}")
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            st.session_state["dyn_recos_preview"] = rows
            if st.button("Apply to Generation"):
                ss = st.session_state
                recs = ss.get("dyn_recos_rows") or ss.get("dyn_recos_preview_rows") or ss.get("dyn_recos") or []
                # ^ use whichever var holds your list of recommendation dicts

                picked_names = []
                weights_map = {}

                for r in (recs or []):
                    name = (r.get("name") or r.get("section") or r.get("title") or "").strip()
                    if not name:
                        continue
                    w = int(r.get("weight") or r.get("priority") or r.get("score") or 0)
                    weights_map[name] = w
                    if w >= 3:   # your threshold
                        picked_names.append(name)

                # 1) Give the Generation page EVERYTHING it needs
                ss["gen_dynamic_recos"] = recs                 # full objects (names, weights, etc.)
                ss["gen_dyn_sel"] = picked_names               # the actual selection
                ss["gen_dyn_weights"] = weights_map            # name -> weight for ordering

                # 2) Prevent the Gen tab’s init from clobbering user picks on first load
                import json, hashlib
                facts = ss.get("rfp_facts") or {}
                facts_hash = hashlib.sha1(json.dumps(facts, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
                ss["gen_last_facts_hash"] = facts_hash
                ss["gen_initialized"] = True

                st.success(f"Applied {len(picked_names)} dynamic sections to Proposal Generation.")


# ----------------------------- Main Render -----------------------------
def render_understanding(
    cfg,
    oai: OpenAI,
    RFP_FACTS_DIR: Path,
    SUPPORTED_UPLOAD_TYPES: List[str],
    TEMPLATE_DIR: Path,
    STATIC_DIR: Path,
    STATIC_MAP_FILE: Path,
    STATIC_SECTIONS: List[str],
):
    st.header("📘 RFP Understanding")
    st.caption("Upload RFPs → Generate Understanding → Value Props → Chat → Export")

    # Upload & init store
    uploaded = st.file_uploader("Upload RFP files", type=SUPPORTED_UPLOAD_TYPES, accept_multiple_files=True)
    if uploaded:
        paths = []
        up_dir = Path("data/uploads"); up_dir.mkdir(parents=True, exist_ok=True)
        for f in uploaded:
            p = up_dir / f.name
            p.write_bytes(f.getvalue())
            paths.append(str(p))
        st.session_state["uploaded_paths"] = paths
        st.success(f"Uploaded {len(paths)} files. Initializing vector store…")
        try:
            store = init_uploaded_store(cfg)
            st.session_state["up_store"] = store
        except Exception as e:
            st.error(f"Vector store init failed: {e}")

    # Generate Understanding
    st.markdown("### 🧠 Generate Understanding")
    st.caption("Extract structured facts from the uploaded RFP using RAG + LLM.")
    col_g1, col_g2 = st.columns([1,3])
    with col_g1:
        do_gen = st.button("🚀 Generate Understanding", type="primary")
    with col_g2:
        add_notes = st.text_input("Optional notes / emphasis (affects retrieval cues)", value="")

    if do_gen:
        store = st.session_state.get("up_store")
        if not store:
            st.warning("Please upload files first — no uploaded vector store available.")
        else:
            prompt = (
                "Context focus cues: contact details; submission instructions; schedule; evaluation; scope; compliance; proposal organization; minimum qualifications\n\n"
                "Using only the provided context, extract an RFP Understanding as strict JSON. Return ONLY a single JSON object, no prose.\n\n"
                f"{SCHEMA_HINT}\n\n"
                f"Notes/Priorities: {add_notes}"
            )
            try:
                out, _ = rag_answer_uploaded(store, oai, cfg, prompt, top_k=12)
            except Exception as e:
                out = f"(LLM/RAG error) {e}"
            try:
                parsed = _coerce_json(out)
                facts = _ensure_dict(parsed)
            except Exception:
                st.error("The model did not return valid JSON. Showing raw output below.")
                st.code(out)
                facts = {}
            if facts:
                sch = facts.get("schedule") or {}
                for k in ["release_date","qna_deadline","addendum_final_date","submission_due","award_target_date","anticipated_start_date"]:
                    if isinstance(sch.get(k), str): sch[k] = {"date": sch[k]}
                for k in ["pre_bid_conference","site_visit"]:
                    v = sch.get(k)
                    if isinstance(v, str): sch[k] = {"datetime": v, "location": ""}
                facts["schedule"] = sch
                st.session_state["rfp_facts"] = facts
                path = save_rfp_facts_json(RFP_FACTS_DIR, facts, "rfp_facts_latest.json")
                st.success(f"Understanding JSON saved → {path.name}")

    # Facts viewer
    facts = st.session_state.get("rfp_facts")
    if isinstance(facts, str):
        try: facts = _ensure_dict(_coerce_json(facts))
        except Exception: facts = {}
    if facts:
        st.markdown("### 📑 Understanding Summary")
        render_rfp_facts(facts)

    # Value props
    st.markdown("---")
    colvp1, colvp2 = st.columns(2)
    with colvp1: _generate_vp_from_sp(cfg, oai)
    with colvp2: _generate_vp_from_web(cfg, oai)

    # Chats
    st.markdown("---")
    tabs = st.tabs(["RFP Chat", "Open-Web Chat"])
    with tabs[0]: _render_rfp_chat(cfg, oai)
    with tabs[1]: _render_web_chat(cfg, oai)

    # Dynamic sections
    st.markdown("---")
    _recommend_dynamic_sections(cfg, oai)

    # Export Understanding
    st.markdown("---")
    st.markdown("### ⬇️ Export Understanding Report (.docx)")
    add_toc = st.checkbox("Add Table of Contents", value=True)
    if st.button("Build & Download"):
        facts = st.session_state.get("rfp_facts") or {}
        
        rfp_msgs = st.session_state.get("rfp_chat_messages", [])
        sp_msgs = st.session_state.get("sp_chat_messages", [])
        try:
            docx_bytes = build_understanding_docx(
                facts,
                rfp_chat_messages=rfp_msgs,
                sp_chat_messages=sp_msgs,
                add_toc=add_toc,
                bct_vp_text=(facts.get("bct_vp_text") if isinstance(facts, dict) else None),
                bct_vp_sources=(facts.get("bct_vp_sources") if isinstance(facts, dict) else None),
                generic_vp_text=(facts.get("generic_vp_text") if isinstance(facts, dict) else None),
            )
            st.download_button(
                "Download .docx",
                data=docx_bytes,
                file_name="understanding_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        except Exception as e:
            st.error(f"Failed to build report: {e}")
