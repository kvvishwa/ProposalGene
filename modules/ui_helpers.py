# modules/ui_helpers.py
# -----------------------------------------------------------------------------
# UI helpers: clipboard buttons, sticky chat CSS, Enter-to-send,
# RFP facts renderer with inline provenance expanders.
# -----------------------------------------------------------------------------

from __future__ import annotations
import json
import streamlit as st
from streamlit.components.v1 import html as comp_html

# --- add near the top of modules/ui_helpers.py (with other helpers) ---
def _as_dateish_dict(val):
    """
    Coerce schedule-like values to a dict so .get() calls won't crash.
    Accepts:
      - dict: return as-is
      - str: wrap as {"date": <str>}
      - None/other: {}
    """
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        return {"date": val}
    return {}


def inject_sticky_chat_css():
    st.markdown("""
    <style>
    .sticky-input { position: sticky; bottom: 0; background: var(--background-color); padding: .5rem .25rem; z-index: 10; }
    </style>
    """, unsafe_allow_html=True)


def copy_to_clipboard_button(text: str, key: str, label: str = "Copy"):
    comp_html(f"""
    <button onclick="navigator.clipboard.writeText({json.dumps(text)})"
            style="padding:.35rem .6rem; border:1px solid #ccc; border-radius:6px; cursor:pointer; margin:.25rem 0;">
        {label}
    </button>
    """, height=40)


def copy_sources_button(sources: list[str], key: str, label: str = "Copy sources"):
    blob = "\n".join(sources or [])
    copy_to_clipboard_button(blob, key=key, label=label)


def enable_enter_to_send(placeholder_text: str):
    comp_html(f"""
    <script>
      const txtAreas = parent.document.querySelectorAll('textarea');
      for (const ta of txtAreas) {{
        ta.addEventListener('keydown', function(e) {{
          if (e.key === 'Enter' && !e.shiftKey) {{
            const forms = parent.document.querySelectorAll('form');
            if (forms && forms.length) {{
              e.preventDefault();
              const btns = forms[0].querySelectorAll('button[type="submit"]');
              if (btns && btns.length) btns[0].click();
            }}
          }}
        }});
      }}
    </script>
    """, height=0)


# ---------------- RFP Facts Renderer ----------------

def _prov_expander(label: str, prov_list: list):
    if prov_list:
        with st.expander(f"🔍 Provenance — {label}", expanded=False):
            for i, p in enumerate(prov_list, 1):
                if isinstance(p, dict):
                    quote = p.get("quote") or p.get("text") or ""
                    src = p.get("source") or p.get("file") or p.get("library") or ""
                    loc = p.get("locator") or p.get("page") or ""
                    st.markdown(f"**{i}.** {src} {('— ' + str(loc)) if loc else ''}")
                    if quote:
                        st.code(quote[:1000], language="text")
                else:
                    st.code(str(p)[:1000], language="text")


def _render_bullets(label: str, items: list | None):
    """Small helper to render a labeled bullet list or a dash if empty."""
    items = items or []
    if items:
        st.write(f"**{label}:**")
        for x in items:
            st.markdown(f"- {x}")
    else:
        st.write(f"**{label}:** —")


def _render_scope_section(facts: dict, show_provenance: bool = False):
    """
    Renders a dedicated 'Scope' section.
    Supports:
      - facts['scope']['in_scope'] / ['out_of_scope']
      - or facts['scope']['scope_raw'] (fallback)
    """
    scope = (facts or {}).get("scope") or {}
    if not scope:
        return  # nothing to show

    st.markdown("#### Scope")
    if "in_scope" in scope or "out_of_scope" in scope:
        _render_bullets("In Scope", scope.get("in_scope"))
        _render_bullets("Out of Scope", scope.get("out_of_scope"))
    elif "scope_raw" in scope:
        _render_bullets("Scope (Detected)", scope.get("scope_raw"))

    # If in future you attach provenance for scope, render like this:
    # if show_provenance:
    #     _prov_expander("Scope", scope.get("provenance") or [])


def render_rfp_facts(facts: dict, show_provenance: bool = False):
    """
    Pretty renders the structured facts into cards/tables.
    If show_provenance=True, adds small expanders with evidence under each section.
    """
    facts = facts or {}
    sol = facts.get("solicitation", {}) or {}
    pocs = facts.get("points_of_contact", []) or []
    sch = facts.get("schedule", {}) or {}
    sub = facts.get("submission_instructions", {}) or {}
    minq = facts.get("minimum_qualifications", {}) or {}
    org = facts.get("proposal_organization", {}) or {}
    evalc = facts.get("evaluation_and_selection", {}) or {}
    terms = facts.get("contract_and_compliance", {}) or {}
    miss = facts.get("missing_notes", []) or []

    st.markdown("#### Solicitation")
    cols = st.columns(2)
    with cols[0]:
        st.write("**Issuing Entity**:", sol.get("issuing_entity_name") or "—")
        st.write("**Dept/Office**:", sol.get("department_or_office") or "—")
        st.write("**Type**:", sol.get("issuing_entity_type") or "—")
    with cols[1]:
        st.write("**Solicitation ID**:", sol.get("solicitation_id") or "—")
        st.write("**Title**:", sol.get("solicitation_title") or "—")
        st.write("**Comms Policy**:", sol.get("communication_policy") or "—")

    st.markdown("#### Points of Contact")
    if pocs:
        for i, p in enumerate(pocs, 1):
            st.markdown(f"- {'**Primary** — ' if p.get('primary') else ''}{p.get('name','')} ({p.get('title','')})  \n"
                        f"  {p.get('email','')} • {p.get('phone','')}  \n"
                        f"  {p.get('address','')}")
    else:
        st.caption("— none detected —")

    if show_provenance:
        _prov_expander("Points of Contact", evalc.get("provenance", []))  # if present under evalc
        # POC provenance not standardized; users often attach at section-level; show if present in facts key:
        if facts.get("poc_provenance"):
            _prov_expander("POC (section-level)", facts["poc_provenance"])

    st.markdown("#### Schedule & Deadlines")
    def _sched_row(label, node_key):
        node = _as_dateish_dict(sch.get(node_key))
        # tolerate alternate keys like 'datetime'/'due'
        date = node.get("date") or node.get("datetime") or node.get("due") or ""
        time = node.get("time") or ""
        tz   = node.get("tz") or node.get("timezone") or ""
        pol  = node.get("delivery_cutoff_policy") or node.get("policy") or ""
        st.write(f"**{label}**: {date} {time} {tz} {('— ' + pol) if pol else ''}")
        if show_provenance:
            _prov_expander(label, node.get("provenance") or [])

    _sched_row("Release Date", "release_date")
    _sched_row("Pre-bid Conference", "pre_bid_conference")
    _sched_row("Site Visit", "site_visit")
    _sched_row("Q&A Deadline", "qna_deadline")
    _sched_row("Addendum Final Date", "addendum_final_date")
    _sched_row("Submission Due", "submission_due")
    _sched_row("Award Target", "award_target_date")
    _sched_row("Anticipated Start", "anticipated_start_date")
    st.write("**Proposal Validity (days):**", sch.get("proposal_validity_days") or "—")
    st.write("**Contract Term:**", sch.get("contract_term") or "—")
    st.write("**Renewal Options:**", sch.get("renewal_options") or "—")

    st.markdown("#### Submission Instructions")
    st.write("**Method**:", sub.get("method") or "—")
    st.write("**Portal**:", f"{sub.get('portal_name','')} {sub.get('portal_url','')}".strip() or "—")
    st.write("**Email:**", sub.get("email_address") or "—")
    st.write("**Physical Address:**", sub.get("physical_address") or "—")
    st.write("**Copies Required:**", sub.get("copies_required") or "—")
    if sub.get("format_requirements"):
        st.write("**Format Requirements:**")
        for x in sub["format_requirements"]:
            st.markdown(f"- {x}")
    st.write("**Labeling:**", sub.get("labeling_instructions") or "—")
    st.write("**Registration:**", sub.get("registration_requirements") or "—")
    st.write("**Late Submissions Policy:**", sub.get("late_submissions_policy") or "—")
    if show_provenance:
        _prov_expander("Submission Instructions", sub.get("provenance") or [])

    st.markdown("#### Minimum Qualifications")
    for label, key in [("Licenses/Certifications","licenses_certifications"),
                       ("Years of Experience","years_experience"),
                       ("Similar Projects","similar_projects"),
                       ("Insurance Requirements","insurance_requirements"),
                       ("Bonding","bonding"),
                       ("Financials","financials"),
                       ("Other Mandatory","other_mandatory")]:
        val = minq.get(key)
        if isinstance(val, list):
            if val:
                st.write(f"**{label}:**")
                for x in val:
                    st.markdown(f"- {x}")
            else:
                st.write(f"**{label}:** —")
        else:
            st.write(f"**{label}:** {val or '—'}")
    if show_provenance:
        _prov_expander("Minimum Qualifications", minq.get("provenance") or [])

    st.markdown("#### Proposal Organization & Forms")
    if org.get("required_sections"):
        st.write("**Required Sections:**")
        for x in org["required_sections"]:
            st.markdown(f"- {x}")
    if org.get("page_limits"):
        st.write("**Page Limits:**")
        for pl in org["page_limits"]:
            st.markdown(f"- {pl.get('section','')}: {pl.get('limit','')}")
    st.write("**Mandatory Forms:**", ", ".join(org.get("mandatory_forms") or []) or "—")
    st.write("**Pricing Format:**", org.get("pricing_format") or "—")
    st.write("**Exceptions Policy:**", org.get("exceptions_policy") or "—")
    if show_provenance:
        _prov_expander("Proposal Organization", org.get("provenance") or [])

    # >>> NEW: Dedicated Scope section (appears before Contract & Compliance)
    _render_scope_section(facts, show_provenance=show_provenance)

    st.markdown("#### Evaluation & Selection")
    crit = evalc.get("criteria") or []
    if crit:
        for c in crit:
            nm = c.get("name",""); wt = c.get("weight",""); desc = c.get("description","")
            st.markdown(f"- **{nm}** ({wt or '—'}): {desc or ''}")
            if show_provenance:
                _prov_expander(f"Criteria — {nm}", c.get("provenance") or [])
    st.write("**Pass/Fail Criteria:**", ", ".join(evalc.get("pass_fail_criteria") or []) or "—")
    for label, key in [("Oral Presentations","oral_presentations"),
                       ("Product Demonstrations","demonstrations"),
                       ("Best & Final Offers","best_and_final_offers"),
                       ("Negotiations","negotiations")]:
        node = evalc.get(key) or {}
        req = node.get("required") if "required" in node else node.get("allowed")
        st.write(f"**{label}:**", f"{req}" if req is not None else "—", node.get("notes",""))
        if show_provenance:
            _prov_expander(label, node.get("provenance") or [])
    st.write("**Selection Process Summary:**", evalc.get("selection_process_summary") or "—")
    st.write("**Protest Procedure:**", evalc.get("protest_procedure") or "—")
    if show_provenance:
        _prov_expander("Evaluation (section)", evalc.get("provenance") or [])

    st.markdown("#### Contract & Compliance")
    for label, arr in [("Key Terms", terms.get("key_terms") or []),
                       ("Security & Privacy", terms.get("security_privacy") or [])]:
        if arr:
            st.write(f"**{label}:**")
            for x in arr:
                st.markdown(f"- {x}")
        else:
            st.write(f"**{label}:** —")
    st.write("**SOW Summary:**", terms.get("sow_summary") or "—")
    if show_provenance:
        _prov_expander("Contract & Compliance", terms.get("provenance") or [])

    if miss:
        st.markdown("#### Missing / Uncertain")
        for m in miss:
            st.markdown(f"- {m}")
