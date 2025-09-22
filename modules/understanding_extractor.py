# modules/understanding_extractor.py
# -----------------------------------------------------------------------------
# Understanding core (hardened)
#  - extract_rfp_facts: first big-pass → targeted sectional passes → merge → notes
#  - facts_to_context:  convert rich facts → lean ProposalContext
#  - infer_section_preferences / plan_dynamic_sections / plan_to_blueprint
#  - Robust JSON parsing and missing-data handling
#  - NEW: Scope mining from uploaded files + heuristic parsing
# -----------------------------------------------------------------------------

from __future__ import annotations

import json as pyjson
import re
import ast
from typing import Dict, List, Tuple, Iterable, Optional

from modules.type import ProposalContext, SectionPlan, GenerationKnobs, DYNAMIC_SECTION_CATALOG
from modules.app_helpers import rag_answer_uploaded

# ------------------------- SCOPE (heuristics) --------------------------------
SCOPE_HEADINGS = [
    r"scope of work",
    r"statement of work",
    r"project scope",
    r"scope",
    r"services? to be provided",
    r"in[-\s]?scope",
    r"out[-\s]?of[-\s]?scope",
    r"deliverables?",
]

_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*)?\s*(?:{})\s*$".format("|".join(SCOPE_HEADINGS)),
    re.IGNORECASE | re.MULTILINE,
)

_NEXT_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*)?\s*[A-Z][A-Za-z0-9\s\-_/()]{3,}\s*$",
    re.MULTILINE,
)

_BULLET_RE = re.compile(r"^\s*([\-–•\*\u2022]|\d+[\.)])\s+")
_EMPTY_LINE_RE = re.compile(r"\n{3,}")


def _normalize(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _EMPTY_LINE_RE.sub("\n\n", text)
    # de-hyphenate common OCR breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text


def _extract_section(text: str, start_idx: int) -> str:
    """Grab text from a heading at start_idx to the next heading (or EOF)."""
    tail = text[start_idx:]
    nxt = _NEXT_HEADING_RE.search(tail, 1)  # skip first line (the current heading)
    if nxt:
        return tail[:nxt.start()].strip()
    return tail.strip()


def _split_bullets(block: str) -> List[str]:
    """Split a block into bullet lines if present; else return paragraph(s)."""
    if not block:
        return []
    lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
    bullet_items: List[str] = []
    current: List[str] = []
    for ln in lines:
        if _BULLET_RE.match(ln):
            if current:
                bullet_items.append(" ".join(current).strip())
                current = []
            bullet_items.append(_BULLET_RE.sub("", ln).strip())
        else:
            current.append(ln)
    if current:
        bullet_items.append(" ".join(current).strip())
    # If no bullets detected, return paragraphs as items (split on blank lines handled above)
    return [it for it in bullet_items if it]


def extract_scope_blocks(full_text: str) -> Dict[str, List[str]]:
    """
    Returns dict with possible keys: 'in_scope', 'out_of_scope', 'scope_raw'
    - Attempts to separate In-Scope vs Out-of-Scope if both appear.
    - Falls back to 'scope_raw' (a list of bullet/paragraph items).
    """
    text = _normalize(full_text or "")
    if not text:
        return {}

    scopes: List[Tuple[str, str]] = []
    for m in _HEADING_RE.finditer(text):
        sect = _extract_section(text, m.start())
        heading = m.group(0).strip().lower()
        scopes.append((heading, sect))

    if not scopes:
        return {}

    in_scope_items: List[str] = []
    out_scope_items: List[str] = []
    mixed_items: List[str] = []

    for heading, block in scopes:
        items = _split_bullets(block)
        if "out" in heading and "scope" in heading:
            out_scope_items.extend(items)
        elif ("in" in heading and "scope" in heading) or ("services to be provided" in heading):
            in_scope_items.extend(items)
        else:
            mixed_items.extend(items)

    # Deduplicate while preserving order, with reasonable caps
    def _dedupe(seq: Iterable[str], cap: int) -> List[str]:
        seen = set()
        out: List[str] = []
        for s in seq:
            if s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) >= cap:
                break
        return out

    result: Dict[str, List[str]] = {}
    if in_scope_items:
        result["in_scope"] = _dedupe(in_scope_items, 120)
    if out_scope_items:
        result["out_of_scope"] = _dedupe(out_scope_items, 120)
    if not result and mixed_items:
        result["scope_raw"] = _dedupe(mixed_items, 150)
    return result


def _mine_scope_from_store(up_store, k: int = 12, max_chars: int = 100_000) -> Dict[str, List[str]]:
    """
    Retrieve likely scope chunks from uploaded store, concatenate, then parse.
    This does not depend on the LLM response and works even if the schema pass is sparse.
    """
    if up_store is None:
        return {}
    seeds = [
        "scope of work", "statement of work", "project scope", "scope",
        "in scope", "out of scope", "deliverables", "services to be provided"
    ]
    texts: List[str] = []
    seen_snips: set[str] = set()
    for s in seeds:
        try:
            docs = up_store.similarity_search(s, k=k) or []
        except Exception:
            docs = []
        for d in docs:
            txt = (getattr(d, "page_content", "") or getattr(d, "content", "") or "")
            if not txt:
                continue
            snip = re.sub(r"\s+", " ", txt[:180]).lower()
            if snip in seen_snips:
                continue
            seen_snips.add(snip)
            texts.append(txt)
            # stop if we already have plenty of text
            if sum(len(t) for t in texts) > max_chars:
                break
        if sum(len(t) for t in texts) > max_chars:
            break

    if not texts:
        return {}
    blob = "\n\n".join(texts)
    return extract_scope_blocks(blob)


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
    # NEW: add scope shell so downstream UI can safely render a section
    data.setdefault("scope", {})
    data.setdefault("missing_notes", [])
    return data


def _is_effectively_empty(data: Dict) -> bool:
    """Check if all top-level sections are empty (excluding missing_notes and scope)."""
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
    try:
        return rag_answer_uploaded(up_store, oai, cfg, prompt, top_k=top_k)
    except Exception as e:
        # Never blow up the extractor if retrieval/model hiccups
        return ("", [f"(rag_error) {e}"])

def extract_rfp_facts(up_store, oai, cfg, return_raw: bool = False):
    """
    First pass: big schema. If effectively empty, run sectional passes with cues.
    Merge all results; append missing_notes for any still-empty sections.
    Also: independently mine SCOPE from the store and attach as a top-level section.
    """
    raw_all: List[str] = []

    # 1) Big schema pass
    raw1, _src1 = _ask(up_store, oai, cfg, _RFP_FACTS_PROMPT, top_k=12)
    raw_all.append("## Big schema pass\n" + (raw1.strip() or "(no model text)"))
    data = _normalize_shell(_safe_json_loads(raw1))

    # 2) If empty, try sectional passes
    if _is_effectively_empty(data):
        for key, sprompt in SECTIONAL_PROMPTS.items():
            cue = RETRIEVAL_CUES.get(key, "")
            section_prompt = f"Context focus cues: {cue}\n\n{sprompt}".strip()
            raw, _src = _ask(up_store, oai, cfg, section_prompt, top_k=14)
            raw_all.append(f"## Sectional pass: {key}\n" + (raw.strip() or "(no model text)"))
            piece = _normalize_shell(_safe_json_loads(raw))
            if piece.get(key):
                data = _merge_dict_deep(data, piece)
    # 3) Attach SCOPE mined directly from uploaded store
    try:
        scope = _mine_scope_from_store(up_store, k=12)
    except Exception:
        scope = {}
    if scope:
        # Add as its own top-level section so UI can render "Scope"
        data["scope"] = scope
        # Optional: if there's no sow_summary, create a short one from in_scope
        if (not (data.get("contract_and_compliance") or {}).get("sow_summary")) and scope.get("in_scope"):
            sample = "; ".join(scope["in_scope"][:5])
            data.setdefault("contract_and_compliance", {}).setdefault("sow_summary", sample[:800])

    # 4) Add missing notes if still blanky
    _add_missing_notes(data)

    raw_combined = "\n\n---\n\n".join([r for r in raw_all if r is not None])
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
# Preference inference, planning, blueprint
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


# --- JSON helpers (robust) ----------------------------------------------------
def _largest_brace_json(s: str) -> str | None:
    """
    Return the largest balanced {...} substring found by scanning with a stack.
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
                    if (best_start == -1) or ((i - start) > (best_end - best_start)):
                        best_start, best_end = start, i
                    start = -1
    if best_start >= 0 and best_end >= 0:
        return s[best_start:best_end+1]
    return None

# REPLACE the old _largest_brace_json with this unified scanner
def _largest_json_block(s: str) -> Optional[str]:
    """
    Return the largest balanced {...} or [...] substring.
    """
    if not s:
        return None
    best = None
    for opens, closes in (("{", "}"), ("[", "]")):
        depth = 0
        start = -1
        for i, ch in enumerate(s):
            if ch == opens:
                if depth == 0:
                    start = i
                depth += 1
            elif ch == closes and depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    cand = s[start:i+1]
                    if best is None or len(cand) > len(best):
                        best = cand
                    start = -1
    return best


# UPDATE _parse_json_lenient to use _largest_json_block and try arrays too
def _parse_json_lenient(text: str) -> Dict:
    """
    Try very hard to parse JSON from LLM output:
    - strip code fences
    - try direct json.loads
    - grab largest balanced {...} or [...]
    - remove trailing commas
    - last resort: ast.literal_eval → re-serialize to strict JSON
    """
    if not text:
        return {}
    raw = _strip_code_fences(text).strip()
    raw = raw.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")

    # 1) direct
    try:
        return pyjson.loads(raw)
    except Exception:
        pass

    # 2) biggest JSON-ish slice
    cand = _largest_json_block(raw) or raw

    try:
        return pyjson.loads(cand)
    except Exception:
        pass

    # 3) kill trailing commas inside both objects & arrays
    cand2 = re.sub(r",(\s*[}\]])", r"\1", cand)
    if cand2 != cand:
        try:
            return pyjson.loads(cand2)
        except Exception:
            pass

    # 4) last resort bridge
    try:
        obj = ast.literal_eval(cand)
        return pyjson.loads(pyjson.dumps(obj))
    except Exception:
        return {}



def _safe_json_loads(s: str) -> Dict:
    """
    Lenient JSON loader used across extractors.
    """
    return _parse_json_lenient(s)


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
        # Prefer strict JSON if the model supports it; fall back otherwise
        res = oai.chat.completions.create(
            model=cfg.ANALYSIS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception:
        res = oai.chat.completions.create(
            model=cfg.ANALYSIS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
    out = (res.choices[0].message.content or "") if hasattr(res, "choices") else ""

    data = _normalize_shell(_safe_json_loads(out))
    if _is_effectively_empty(data):
        _add_missing_notes(data)

    return (data, out) if return_raw else data


# --- Deterministic miners -----------------------------------------------------
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
    found: List[dict] = []
    email_rx = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
    phone_rx = re.compile(r"(?:\+?\d[\d\-\s().]{7,}\d)")
    name_rx  = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")  # crude

    for s in seeds:
        try:
            docs = store.similarity_search(s, k=k) or []
        except Exception:
            docs = []
        for d in docs:
            txt = (getattr(d, "page_content", "") or getattr(d, "content", "") or "")[:1200]
            if not txt or not txt.strip():
                continue
            sig = re.sub(r"\s+", " ", txt[:200]).lower()
            if sig in seen_sig:
                continue
            seen_sig.add(sig)

            emails = email_rx.findall(txt)
            phones = phone_rx.findall(txt)
            # Guess a name near 'contact' or before an email occurrence
            name = ""
            m = re.search(r"(?:contact|communications|to:)\s*[:\-]?\s*(.+)$", txt, re.I | re.M)
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
            if entry(["email"] or entry["phone"]):
                found.append(entry)

    # de-dupe by (email, phone)
    uniq: List[dict] = []
    seen = set()
    for e in found:
        key = (e["email"], e["phone"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    return uniq[:4]
