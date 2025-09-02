# modules/understanding_extractor.py
# -----------------------------------------------------------------------------
# Understanding core (hardened)
#  - extract_rfp_facts: first big-pass → targeted sectional passes → merge → notes
#  - facts_to_context:  convert rich facts → lean ProposalContext
#  - infer_section_preferences / plan_dynamic_sections / plan_to_blueprint
#  - Robust JSON parsing and missing-data handling
# -----------------------------------------------------------------------------

from __future__ import annotations

import json as pyjson
import re
from typing import Dict, List, Tuple

from modules.type import ProposalContext, SectionPlan, GenerationKnobs, DYNAMIC_SECTION_CATALOG
from modules.app_helpers import rag_answer_uploaded


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
_RFP_TYPE_CANON = {
    "Technical Solution": ["technical", "solution", "implementation", "delivery"],
    "Staff Augmentation": ["staff augmentation", "resourcing", "it staff", "augmentation"],
    "Managed Services": ["managed service", "operations", "support", "sustainment"],
    "Consulting & Advisory": ["consulting", "advisory", "assessment", "roadmap"],
    "Off-the-Shelf Implementation": ["cots", "off-the-shelf", "package", "product implementation"],
    "Product/Hardware Procurement": ["hardware", "procurement", "supply", "equipment"],
    "Marketing & Creative Services": ["marketing", "creative", "branding", "campaign"],
    "Construction / Facilities": ["construction", "build", "facilities", "fit-out"],
    "Research & Innovation": ["research", "pilot", "innovation", "prototype"],
}

def _canonicalize_rfp_type(text: str) -> str:
    s = (text or "").lower()
    for canon, cues in _RFP_TYPE_CANON.items():
        if any(cue in s for cue in cues):
            return canon
    return "Technical Solution"

def _strip_code_fences(s: str) -> str:
    if not s:
        return s
    # remove ```json ... ``` or ``` ... ```
    return re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", s, flags=re.IGNORECASE)

def _normalize_shell(data: Dict) -> Dict:
    """Ensure top-level keys exist so UI doesn't KeyError."""
    if not isinstance(data, dict):
        data = {}
    data.setdefault("solicitation", {})
    data.setdefault("points_of_contact", [])
    data.setdefault("schedule", {})
    data.setdefault("submission_instructions", {})
    data.setdefault("minimum_qualifications", {})
    data.setdefault("proposal_organization", {})
    data.setdefault("evaluation_and_selection", {})
    data.setdefault("contract_and_compliance", {})
    data.setdefault("missing_notes", [])
    return data

def _is_effectively_empty(data: Dict) -> bool:
    """Check if all top-level sections are empty."""
    if not isinstance(data, dict) or not data:
        return True
    keys = [
        "solicitation","points_of_contact","schedule","submission_instructions",
        "minimum_qualifications","proposal_organization","evaluation_and_selection",
        "contract_and_compliance"
    ]
    for k in keys:
        v = data.get(k)
        if isinstance(v, dict) and len(v) > 0:
            return False
        if isinstance(v, list) and len(v) > 0:
            return False
    return True

def _merge_dict_deep(base: Dict, add: Dict) -> Dict:
    """Shallow/deep merge: dicts merged, lists concatenated, scalars replaced if add has value."""
    if not isinstance(add, dict):
        return base
    for k, v in add.items():
        if k not in base:
            base[k] = v
            continue
        if isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = _merge_dict_deep(base[k], v)
        elif isinstance(base[k], list) and isinstance(v, list):
            # extend with new items avoiding trivial duplicates
            seen = set(pyjson.dumps(x, sort_keys=True) for x in base[k])
            for item in v:
                sig = pyjson.dumps(item, sort_keys=True)
                if sig not in seen:
                    base[k].append(item)
                    seen.add(sig)
        else:
            # replace only if the new value is truthy / non-empty
            if v not in (None, "", [], {}):
                base[k] = v
    return base

def _add_missing_notes(data: Dict) -> None:
    notes = data.get("missing_notes") or []
    def add(msg: str):
        if msg and msg not in notes:
            notes.append(msg)

    if not data.get("solicitation"):
        add("No solicitation header (issuer/title/id) found in provided excerpts.")
    if not data.get("points_of_contact"):
        add("No points of contact detected.")
    if not data.get("schedule"):
        add("No schedule/deadlines found.")
    if not data.get("submission_instructions"):
        add("No submission/format instructions detected.")
    if not data.get("minimum_qualifications"):
        add("No minimum qualifications found.")
    if not data.get("proposal_organization"):
        add("No proposal organization/page limits/forms detected.")
    if not data.get("evaluation_and_selection"):
        add("No evaluation/selection criteria detected.")
    if not data.get("contract_and_compliance"):
        add("No contract/compliance terms detected.")
    data["missing_notes"] = notes


# -----------------------------------------------------------------------------
# Big schema prompt (first pass)
# -----------------------------------------------------------------------------
_RFP_FACTS_PROMPT = """
You are extracting procurement facts from an uploaded Request for Proposal (RFP) corpus.

You will be given RFP context (snippets from one or more files). Using ONLY the provided context, return a SINGLE JSON object strictly matching the schema below (no markdown fences, no commentary). If a field is unknown, use an empty string "" or an empty list [] and include a short reason in the "missing_notes" array.

IMPORTANT RULES
- DO NOT hallucinate or infer beyond the text.
- Prefer the most recent/addendum content if conflicting.
- Normalize dates to ISO 8601 (YYYY-MM-DD). Datetimes as YYYY-MM-DDTHH:MM (24h). If a time zone is explicitly stated, preserve it in "tz"; otherwise tz = null.
- Keep labels EXACT as they appear (e.g., 30 pages, 25 points, 35%).
- If multiple values exist (e.g., two POCs), include all in arrays but set a "primary": true on the main one when indicated (e.g., “All communications to …”).
- Always include PROVENANCE: for each key piece you extract, attach a short evidence object with a quote/snippet and a source hint (filename or library label) and any locator (page #, section heading) if present in the text.

OUTPUT SCHEMA (return exactly one JSON object):
{
  "solicitation": {
    "issuing_entity_name": "",
    "issuing_entity_type": "",
    "department_or_office": "",
    "solicitation_id": "",
    "solicitation_title": "",
    "communication_policy": ""
  },
  "points_of_contact": [
    {
      "primary": true,
      "name": "",
      "title": "",
      "email": "",
      "phone": "",
      "address": "",
      "notes": ""
    }
  ],
  "schedule": {
    "release_date": {"date": "", "tz": null, "provenance": []},
    "pre_bid_conference": {
      "required": null, "virtual": null,
      "datetime": {"date": "", "time": "", "tz": null},
      "location": "", "provenance": []
    },
    "site_visit": {
      "required": null,
      "datetime": {"date": "", "time": "", "tz": null},
      "location": "", "provenance": []
    },
    "qna_deadline": {"date": "", "time": "", "tz": null, "provenance": []},
    "addendum_final_date": {"date": "", "provenance": []},
    "submission_due": {"date": "", "time": "", "tz": null, "delivery_cutoff_policy": "", "provenance": []},
    "award_target_date": {"date": "", "provenance": []},
    "anticipated_start_date": {"date": "", "provenance": []},
    "proposal_validity_days": "",
    "contract_term": "",
    "renewal_options": ""
  },
  "submission_instructions": {
    "method": "",
    "portal_name": "",
    "portal_url": "",
    "email_address": "",
    "physical_address": "",
    "copies_required": "",
    "format_requirements": [""],
    "labeling_instructions": "",
    "registration_requirements": "",
    "late_submissions_policy": "",
    "provenance": []
  },
  "minimum_qualifications": {
    "licenses_certifications": [""],
    "years_experience": "",
    "similar_projects": "",
    "insurance_requirements": [""],
    "bonding": [""],
    "financials": "",
    "other_mandatory": [""],
    "provenance": []
  },
  "proposal_organization": {
    "required_sections": [""],
    "page_limits": [{"section": "", "limit": ""}],
    "mandatory_forms": [""],
    "pricing_format": "",
    "exceptions_policy": "",
    "provenance": []
  },
  "evaluation_and_selection": {
    "criteria": [
      {"name": "", "weight": "", "description": "", "provenance": []}
    ],
    "pass_fail_criteria": [""],
    "oral_presentations": {"required": null, "notes": "", "provenance": []},
    "demonstrations": {"required": null, "notes": "", "provenance": []},
    "best_and_final_offers": {"allowed": null, "notes": "", "provenance": []},
    "negotiations": {"allowed": null, "notes": "", "provenance": []},
    "selection_process_summary": "",
    "protest_procedure": "",
    "provenance": []
  },
  "contract_and_compliance": {
    "key_terms": [""],
    "security_privacy": [""],
    "sow_summary": "",
    "provenance": []
  },
  "missing_notes": [""]
}

Respond with the JSON object ONLY.
""".strip()


# -----------------------------------------------------------------------------
# Sectional prompts (fallbacks)
# -----------------------------------------------------------------------------
SECTIONAL_PROMPTS: Dict[str, str] = {
    "solicitation": """
From the context, extract ONLY the following JSON for solicitation header (no commentary):
{
  "solicitation": {
    "issuing_entity_name": "",
    "issuing_entity_type": "",
    "department_or_office": "",
    "solicitation_id": "",
    "solicitation_title": "",
    "communication_policy": ""
  }
}
""",
    "points_of_contact": """
Extract points of contact. Return ONLY JSON:
{
  "points_of_contact": [
    {"primary": true, "name": "", "title": "", "email": "", "phone": "", "address": "", "notes": ""}
  ]
}
Include as many POCs as present; mark primary when indicated.
""",
    "schedule": """
Extract schedule & deadlines. Return ONLY JSON:
{
  "schedule": {
    "release_date": {"date": "", "tz": null, "provenance": []},
    "pre_bid_conference": {"required": null, "virtual": null, "datetime": {"date": "", "time": "", "tz": null}, "location": "", "provenance": []},
    "site_visit": {"required": null, "datetime": {"date": "", "time": "", "tz": null}, "location": "", "provenance": []},
    "qna_deadline": {"date": "", "time": "", "tz": null, "provenance": []},
    "addendum_final_date": {"date": "", "provenance": []},
    "submission_due": {"date": "", "time": "", "tz": null, "delivery_cutoff_policy": "", "provenance": []},
    "award_target_date": {"date": "", "provenance": []},
    "anticipated_start_date": {"date": "", "provenance": []},
    "proposal_validity_days": "",
    "contract_term": "",
    "renewal_options": ""
  }
}
""",
    "submission_instructions": """
Extract submission instructions. Return ONLY JSON:
{
  "submission_instructions": {
    "method": "",
    "portal_name": "",
    "portal_url": "",
    "email_address": "",
    "physical_address": "",
    "copies_required": "",
    "format_requirements": [""],
    "labeling_instructions": "",
    "registration_requirements": "",
    "late_submissions_policy": "",
    "provenance": []
  }
}
""",
    "minimum_qualifications": """
Extract minimum qualifications / mandatory requirements. Return ONLY JSON:
{
  "minimum_qualifications": {
    "licenses_certifications": [""],
    "years_experience": "",
    "similar_projects": "",
    "insurance_requirements": [""],
    "bonding": [""],
    "financials": "",
    "other_mandatory": [""],
    "provenance": []
  }
}
""",
    "proposal_organization": """
Extract proposal organization, page limits and mandatory forms. Return ONLY JSON:
{
  "proposal_organization": {
    "required_sections": [""],
    "page_limits": [{"section": "", "limit": ""}],
    "mandatory_forms": [""],
    "pricing_format": "",
    "exceptions_policy": "",
    "provenance": []
  }
}
""",
    "evaluation_and_selection": """
Extract evaluation & selection criteria. Return ONLY JSON:
{
  "evaluation_and_selection": {
    "criteria": [{"name": "", "weight": "", "description": "", "provenance": []}],
    "pass_fail_criteria": [""],
    "oral_presentations": {"required": null, "notes": "", "provenance": []},
    "demonstrations": {"required": null, "notes": "", "provenance": []},
    "best_and_final_offers": {"allowed": null, "notes": "", "provenance": []},
    "negotiations": {"allowed": null, "notes": "", "provenance": []},
    "selection_process_summary": "",
    "protest_procedure": "",
    "provenance": []
  }
}
""",
    "contract_and_compliance": """
Extract contract & compliance terms. Return ONLY JSON:
{
  "contract_and_compliance": {
    "key_terms": [""],
    "security_privacy": [""],
    "sow_summary": "",
    "provenance": []
  }
}
""",
}

# retrieval cue per section to steer the vector store towards right chunks
RETRIEVAL_CUES: Dict[str, str] = {
    "solicitation": "issuing entity, solicitation ID, title, communication policy",
    "points_of_contact": "contact person, email, phone, procurement office contact",
    "schedule": "schedule, due date, deadlines, pre-bid, site visit, Q&A, addendum, award",
    "submission_instructions": "submission instructions, method, portal, copies, labeling, registration, format",
    "minimum_qualifications": "minimum qualifications, mandatory requirements, insurance, bonding, certifications",
    "proposal_organization": "proposal format, organization, page limits, required sections, forms, pricing format",
    "evaluation_and_selection": "evaluation criteria, weighting, selection process, BAFO, negotiations, protest",
    "contract_and_compliance": "contract terms, security, privacy, compliance, scope of work",
}


# -----------------------------------------------------------------------------
# Extractor (hardened)
# -----------------------------------------------------------------------------
def _ask(up_store, oai, cfg, prompt: str, top_k: int = 12) -> Tuple[str, List[str]]:
    return rag_answer_uploaded(up_store, oai, cfg, prompt, top_k=top_k)

def extract_rfp_facts(up_store, oai, cfg, return_raw: bool = False):
    """
    First pass: big schema. If effectively empty, run sectional passes with cues.
    Merge all results; append missing_notes for any still-empty sections.
    """
    raw_all = []

    # 1) Big schema pass
    raw1, _src1 = _ask(up_store, oai, cfg, _RFP_FACTS_PROMPT, top_k=12)
    raw_all.append(raw1 or "")
    data = _normalize_shell(_safe_json_loads(raw1))

    # 2) If empty, try sectional passes
    if _is_effectively_empty(data):
        for key, sprompt in SECTIONAL_PROMPTS.items():
            cue = RETRIEVAL_CUES.get(key, "")
            section_prompt = f"Context focus cues: {cue}\n\n{sprompt}".strip()
            raw, _src = _ask(up_store, oai, cfg, section_prompt, top_k=14)
            raw_all.append(raw or "")
            piece = _normalize_shell(_safe_json_loads(raw))
            if piece.get(key):
                data = _merge_dict_deep(data, piece)

    # 3) Add missing notes if still blanky
    _add_missing_notes(data)

    raw_combined = "\n\n---\n\n".join([r for r in raw_all if r])
    return (data, raw_combined) if return_raw else data


# -----------------------------------------------------------------------------
# Convert facts → lean context
# -----------------------------------------------------------------------------
def facts_to_context(facts: dict) -> ProposalContext:
    sol = facts.get("solicitation", {}) or {}
    sched = facts.get("schedule", {}) or {}
    due = (sched.get("submission_due") or {}).get("date") or None

    issuer = sol.get("issuing_entity_name") or ""
    title = sol.get("solicitation_title") or sol.get("solicitation_id") or ""
    terms = facts.get("contract_and_compliance", {}) or {}
    evalc = facts.get("evaluation_and_selection", {}) or {}
    org = facts.get("proposal_organization", {}) or {}

    summary_bits = []
    if issuer or title:
        summary_bits.append(f"This RFP ('{title}') is issued by {issuer}." if issuer else f"This RFP is titled '{title}'.")
    if due:
        summary_bits.append(f"Proposals are due by {due}.")
    if org.get("required_sections"):
        summary_bits.append(f"It specifies required sections such as {', '.join(org['required_sections'][:4])}{'...' if len(org['required_sections'])>4 else ''}.")
    crit = evalc.get("criteria") or []
    if crit:
        summary_bits.append(f"Evaluation includes {', '.join([c.get('name','').strip() for c in crit if c.get('name')][:3])}{'...' if len(crit)>3 else ''}.")
    if terms.get("key_terms"):
        summary_bits.append(f"Key contractual terms noted include {', '.join((terms['key_terms'] or [])[:3])}{'...' if len(terms.get('key_terms') or [])>3 else ''}.")
    summary = " ".join([s for s in summary_bits if s]).strip()

    ctx = ProposalContext(
        customer_name=issuer or "",
        opportunity_title=title or "",
        rfp_type="Technical Solution",
        due_date=due,
        summary=summary,
        constraints=[],
        mandatory_sections=org.get("required_sections") or [],
        preferred_sections={},
        notes=[]
    )
    ctx.preferred_sections = infer_section_preferences(ctx)
    return ctx


# -----------------------------------------------------------------------------
# Preference inference, planning, blueprint (unchanged)
# -----------------------------------------------------------------------------
def infer_section_preferences(ctx: ProposalContext) -> Dict[str, float]:
    text = f"{ctx.summary} {' '.join(ctx.constraints)} {' '.join(ctx.mandatory_sections)}".lower()
    base = {label: 0.4 for label in DYNAMIC_SECTION_CATALOG}

    for m in ctx.mandatory_sections:
        mm = m.lower()
        for label in DYNAMIC_SECTION_CATALOG:
            if label.lower() in mm:
                base[label] = max(base[label], 0.9)

    rtype = ctx.rfp_type
    if rtype == "Staff Augmentation":
        for k in ("Staffing Plan & Roles", "Approach & Methodology", "Transition & Knowledge Transfer"):
            base[k] = max(base.get(k, 0.4), 0.8)
    elif rtype == "Managed Services":
        for k in ("Service Levels & Governance", "Quality Management", "Project Governance"):
            base[k] = max(base.get(k, 0.4), 0.8)
    elif rtype == "Consulting & Advisory":
        for k in ("Approach & Methodology", "Risk & Mitigation Plan", "Change Management"):
            base[k] = max(base.get(k, 0.4), 0.75)

    kw_map = {
        "sla": "Service Levels & Governance",
        "service level": "Service Levels & Governance",
        "governance": "Project Governance",
        "risk": "Risk & Mitigation Plan",
        "transition": "Transition & Knowledge Transfer",
        "knowledge transfer": "Transition & Knowledge Transfer",
        "timeline": "Timeline & Milestones",
        "schedule": "Timeline & Milestones",
        "quality": "Quality Management",
        "security": "Compliance & Security",
        "compliance": "Compliance & Security",
        "roles": "Staffing Plan & Roles",
        "staffing": "Staffing Plan & Roles",
        "approach": "Approach & Methodology",
        "methodology": "Approach & Methodology",
        "assumption": "Assumptions & Exclusions",
        "exclusion": "Assumptions & Exclusions",
        "change": "Change Management",
    }
    for kw, label in kw_map.items():
        if kw in text:
            base[label] = min(1.0, base.get(label, 0.4) + 0.15)

    for k in base:
        base[k] = max(0.0, min(1.0, base[k]))
    return base


def plan_dynamic_sections(ctx: ProposalContext, static_map_now: Dict[str, str], top_n: int = 6) -> List[SectionPlan]:
    items = sorted(ctx.preferred_sections.items(), key=lambda kv: kv[1], reverse=True)
    plans: List[SectionPlan] = []
    for label, weight in items:
        if static_map_now.get(label):
            continue
        if label not in DYNAMIC_SECTION_CATALOG:
            continue
        plans.append(SectionPlan(
            label=label,
            priority=float(weight),
            profile_key=label,
            knobs=GenerationKnobs(
                top_k=6, style="bullets", tone="Professional",
                max_bullets=6, target_words=180, dynamic_heading_level=1,
                force_title=True, include_sources=True
            ),
            anchor_hint=None, static_conflict=False
        ))
        if len(plans) >= top_n:
            break
    return plans


def plan_to_blueprint(plans: List[SectionPlan], current_template_name: str | None, tone: str = "Professional") -> Dict:
    dyn_sel = [p.label for p in plans]
    return {
        "template": current_template_name or "— choose a template —",
        "static_sel": None,
        "dyn_sel": dyn_sel,
        "include_sources": True,
        "top_k": 6,
        "tone": tone,
        "use_anchors": True,
        "add_headings": False,
        "rec_style": "bullets",
        "per_section_k": 0,
        "add_toc": True,
        "page_breaks": True,
        "template_has_headings": True,
    }

# --- add this helper (no extra imports needed) ---
def _largest_brace_json(s: str) -> str | None:
    """
    Return the largest balanced {...} substring found by scanning with a stack.
    This avoids PCRE recursion (?R) and works with Python's 're' limitations.
    """
    if not s:
        return None
    best_start = best_end = -1
    depth = 0
    start = -1
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    # candidate complete object
                    if (best_start == -1) or ((i - start) > (best_end - best_start)):
                        best_start, best_end = start, i
                    start = -1
    if best_start >= 0 and best_end >= 0:
        return s[best_start:best_end+1]
    return None

def _safe_json_loads(s: str) -> Dict:
    """
    Try direct JSON load after stripping code fences. If that fails,
    extract the largest balanced {...} block with a stack scanner and load that.
    """
    if not s:
        return {}
    s2 = _strip_code_fences(s).strip()
    # 1) direct attempt
    try:
        return pyjson.loads(s2)
    except Exception:
        pass
    # 2) try the largest balanced {...} region
    cand = _largest_brace_json(s2)
    if cand:
        try:
            return pyjson.loads(cand)
        except Exception:
            pass
    # 3) give up
    return {}


def extract_rfp_facts_from_raw_text(raw_text: str, oai, cfg, return_raw: bool = False):
    """
    Direct (non-RAG) extraction: pass a large raw text blob to the model.
    Useful when vector store is empty or retrieval returns nothing.
    """
    from modules.understanding_extractor import (
        _RFP_FACTS_PROMPT, _normalize_shell, _safe_json_loads, _is_effectively_empty, _add_missing_notes
    )
    if not raw_text or not raw_text.strip():
        data = _normalize_shell({})
        _add_missing_notes(data)
        return (data, "") if return_raw else data

    prompt = (
        "You are extracting procurement facts from the following RFP text. "
        "Use ONLY this text; do not hallucinate. Return a SINGLE JSON object matching the schema.\n\n"
        f"RFP TEXT (may be partial):\n{raw_text[:120000]}\n\n"
        f"{_RFP_FACTS_PROMPT}"
    )
    try:
        res = oai.chat.completions.create(
            model=cfg.ANALYSIS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        out = res.choices[0].message.content or ""
    except Exception:
        out = ""

    data = _normalize_shell(_safe_json_loads(out))
    if _is_effectively_empty(data):
        _add_missing_notes(data)

    return (data, out) if return_raw else data



def _mine_pocs_from_store(store, k: int = 8) -> List[dict]:
    """
    Deterministic POC miner: pull likely contact snippets and extract emails/phones.
    Returns a list of {"name":?, "email":?, "phone":?, "notes":?, "primary": False}
    """
    seeds = [
        "contact email", "point of contact", "all communications", "procurement contact",
        "questions concerning", "rfp contact", "issuing office contact"
    ]
    seen_sig = set()
    found = []
    email_rx = _re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", _re.I)
    phone_rx = _re.compile(r"(?:\+?\d[\d\-\s().]{7,}\d)")
    name_rx  = _re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")  # crude

    for s in seeds:
        try:
            docs = store.similarity_search(s, k=k) or []
        except Exception:
            docs = []
        for d in docs:
            txt = (getattr(d, "page_content", "") or getattr(d, "content", "") or "")[:1200]
            if not txt or not txt.strip():
                continue
            sig = _re.sub(r"\s+", " ", txt[:200]).lower()
            if sig in seen_sig:
                continue
            seen_sig.add(sig)

            emails = email_rx.findall(txt)
            phones = phone_rx.findall(txt)
            # Guess a name near 'contact' or before an email occurrence
            name = ""
            m = _re.search(r"(?:contact|communications|to:)\s*[:\-]?\s*(.+)$", txt, _re.I | _re.M)
            if m:
                line = m.group(1)[:120]
                nm = name_rx.search(line)
                if nm:
                    name = nm.group(1).strip()
            if not name and emails:
                # look backwards a bit from the first email
                pos = txt.find(emails[0])
                snip = txt[max(0, pos-80):pos]
                nm = name_rx.search(snip)
                if nm:
                    name = nm.group(1).strip()

            entry = {
                "primary": False,
                "name": name,
                "title": "",
                "email": emails[0] if emails else "",
                "phone": phones[0] if phones else "",
                "address": "",
                "notes": "mined from keyword context" if (emails or phones) else "",
            }
            if entry["email"] or entry["phone"]:
                found.append(entry)

    # de-dupe by (email, phone)
    uniq = []
    seen = set()
    for e in found:
        key = (e["email"], e["phone"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    return uniq[:4]
