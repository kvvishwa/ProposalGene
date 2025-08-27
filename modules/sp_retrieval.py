# modules/sp_retrieval.py
# -----------------------------------------------------------------------------
# SharePoint retrieval upgraded to structured evidence packs.
# Now also supports:
#   - get_sp_evidence_for_question(question, facts, cfg, k=8)
#   - get_store_evidence(store, question, facts, k=8)
# These are used by Chat (SP) and Chat (RFP Understanding) respectively.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re

from modules.vectorstore import init_sharepoint_store
from modules.app_helpers import _similarity_search  # lightweight wrapper around store.similarity_search

# Section cue words (used by dynamic generation)
_SECTION_CUES: Dict[str, str] = {
    "Executive Summary": "objectives outcomes value differentiators benefits similar wins impact",
    "Approach & Methodology": "approach methodology phases activities deliverables tools work plan",
    "Staffing Plan & Roles": "staffing roles responsibilities team structure key personnel coverage hours",
    "Transition & Knowledge Transfer": "transition onboarding ramp-up knowledge transfer shadowing KT plan",
    "Service Levels & Governance": "service levels SLA KPI reporting cadence governance escalation",
    "Assumptions & Exclusions": "assumptions exclusions constraints dependencies",
    "Risk & Mitigation Plan": "risk risks mitigation register probability impact response",
    "Project Governance": "governance steering committee decision-making reporting escalation cadence",
    "Change Management": "change management adoption training communications readiness",
    "Quality Management": "quality assurance QA QC testing metrics continuous improvement",
    "Compliance & Security": "compliance security privacy standards certifications data protection",
    "Timeline & Milestones": "timeline milestones schedule plan deliverables",
}

_SECTION_PREFER_TYPES: Dict[str, List[str]] = {
    "Executive Summary": ["case_study", "offering", "overview"],
    "Approach & Methodology": ["methodology", "process", "playbook"],
    "Staffing Plan & Roles": ["staffing", "roles", "organization", "resume", "profile"],
    "Transition & Knowledge Transfer": ["transition", "onboarding"],
    "Service Levels & Governance": ["sla", "governance", "operations"],
    "Assumptions & Exclusions": ["assumptions", "contract", "sow"],
    "Risk & Mitigation Plan": ["risk", "governance"],
    "Project Governance": ["governance", "operations"],
    "Change Management": ["change", "adoption", "training"],
    "Quality Management": ["quality", "testing"],
    "Compliance & Security": ["security", "compliance", "policy"],
    "Timeline & Milestones": ["plan", "schedule", "project"],
}


@dataclass
class Evidence:
    text: str
    source: str
    score: float
    why_relevant: str = ""
    page_hint: Optional[str] = None


# ----------------------------- utilities -----------------------------

def _top_terms(text: str, max_terms: int = 8) -> List[str]:
    """Naive keyword picker: longest distinct words excluding stop words."""
    stop = set("the a an and or for with from into to of we you they our your is are be by as on at in".split())
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)
    words = [w.lower() for w in words if w.lower() not in stop]
    words = sorted(set(words), key=lambda w: (-len(w), w))
    return words[:max_terms]


def _extract_facts_terms(facts: Optional[dict]) -> Tuple[List[str], List[str]]:
    """Pull a few high-signal terms from RFP facts: issuer + keywords."""
    issuers: List[str] = []
    keywords: List[str] = []
    if not isinstance(facts, dict):
        return issuers, keywords

    sol = facts.get("solicitation") or {}
    issuer = sol.get("issuer") or sol.get("agency") or sol.get("entity") or ""
    if issuer:
        issuers.append(str(issuer))

    # scrape a few blobs for keywords
    for key in ("proposal_organization", "minimum_qualifications", "evaluation_and_selection", "submission_instructions"):
        blob = facts.get(key) or {}
        if isinstance(blob, dict):
            for v in blob.values():
                if isinstance(v, str):
                    keywords.extend(_top_terms(v, max_terms=8))
        elif isinstance(blob, list):
            for v in blob:
                if isinstance(v, str):
                    keywords.extend(_top_terms(v, max_terms=8))

    # dedupe & trim
    seen = set()
    out_kw = []
    for t in keywords:
        t = t.strip().lower()
        if len(t) >= 3 and t not in seen:
            seen.add(t)
            out_kw.append(t)
        if len(out_kw) >= 12:
            break

    return issuers[:2], out_kw[:12]


def _score_snippet(section_or_q: str, text: str, meta: Dict[str, Any], prefer_types: Optional[List[str]]) -> float:
    """Lightweight score using cue words, doc_type, and length."""
    score = 0.0
    # tokens from section or question string
    for token in _top_terms(section_or_q, max_terms=10):
        if token in text.lower():
            score += 0.5

    doc_type = (meta.get("doc_type") or meta.get("doctype") or meta.get("type") or "").lower()
    if prefer_types:
        for pt in prefer_types:
            if pt in doc_type:
                score += 1.0

    n = len(text)
    if 300 <= n <= 1200:
        score += 0.5
    elif n > 1600:
        score -= 0.2
    return score


# ----------------------- planners & retrievers -----------------------

def plan_sp_subqueries(section_label: str, facts: Optional[dict]) -> List[str]:
    """Build 3–5 subqueries combining section cues + issuer + scope keywords."""
    cues = _SECTION_CUES.get(section_label, section_label)
    issuers, kw = _extract_facts_terms(facts)
    base = f"{section_label} {cues}"

    queries = [base]
    if issuers:
        queries.append(f"{base} {issuers[0]}")
    if kw:
        queries.append(f"{base} " + " ".join(kw[:4]))
    if len(kw) > 4:
        queries.append(f"{base} " + " ".join(kw[4:8]))
    if issuers and len(kw) > 8:
        queries.append(f"{base} {issuers[0]} " + " ".join(kw[8:12]))

    # unique, non-empty
    uniq, seen = [], set()
    for q in queries:
        qn = q.strip()
        if qn and qn not in seen:
            uniq.append(qn); seen.add(qn)
    return uniq[:5]


def plan_qa_subqueries(question: str, facts: Optional[dict]) -> List[str]:
    """Plan subqueries for general Q&A (chat)."""
    issuers, kw = _extract_facts_terms(facts)
    base_terms = " ".join(_top_terms(question, max_terms=10)) or question
    queries = [base_terms]
    if issuers:
        queries.append(f"{base_terms} {issuers[0]}")
    if kw:
        queries.append(base_terms + " " + " ".join(kw[:4]))
    if len(kw) > 4:
        queries.append(base_terms + " " + " ".join(kw[4:8]))
    if issuers and len(kw) > 8:
        queries.append(f"{base_terms} {issuers[0]} " + " ".join(kw[8:12]))

    uniq, seen = [], set()
    for q in queries:
        qn = q.strip()
        if qn and qn not in seen:
            uniq.append(qn); seen.add(qn)
    return uniq[:5]


def _gather(pool: List[Tuple[Evidence, float]], subqueries: List[str], section_or_q: str,
            search_fn, k: int, prefer_types: Optional[List[str]]) -> List[Evidence]:
    """Shared gatherer for SP or generic store."""
    seen = set()
    for q in subqueries:
        docs = search_fn(q, k=max(6, k))
        for d in docs or []:
            txt = (getattr(d, "page_content", None) or getattr(d, "content", "") or "").strip()
            if not txt:
                continue
            meta = getattr(d, "metadata", {}) or {}
            src = meta.get("source") or meta.get("file") or ""
            page = meta.get("page") or meta.get("page_number") or meta.get("pageno")
            page_hint = f"p.{page}" if page else None

            sig = (src + "|" + txt[:160]).lower()
            if sig in seen:
                continue
            seen.add(sig)

            sc = _score_snippet(section_or_q, txt, meta, prefer_types)
            why = ""
            doc_type = (meta.get("doc_type") or "").lower()
            if prefer_types and any(pt in doc_type for pt in prefer_types):
                why = f"Matches preferred type: {doc_type}"
            elif "sla" in txt.lower() or "kpi" in txt.lower():
                why = "Contains SLA/KPI language"
            elif "role" in txt.lower() or "responsibilit" in txt.lower():
                why = "Mentions roles/responsibilities"

            pool.append((Evidence(text=txt[:1200], source=src, score=sc, why_relevant=why, page_hint=page_hint), sc))
    # rank and diversify by source
    pool.sort(key=lambda x: x[1], reverse=True)
    ranked = [ev for ev, _ in pool]
    final: List[Evidence] = []
    by_src = set()
    for ev in ranked:
        key = ev.source or ""
        if key and key not in by_src:
            final.append(ev); by_src.add(key)
        elif not key:
            final.append(ev)
        if len(final) >= k:
            break
    return final


def get_sp_evidence(section_label: str, facts: Optional[dict], cfg, k: int = 8, prefer_types: Optional[List[str]] = None) -> List[Evidence]:
    """Evidence pack for a dynamic section (SharePoint store)."""
    try:
        vs = init_sharepoint_store(cfg)
    except Exception:
        return []
    subqueries = plan_sp_subqueries(section_label, facts)
    prefer = prefer_types or _SECTION_PREFER_TYPES.get(section_label, [])
    return _gather([], subqueries, section_label, lambda q, k: _similarity_search(vs, q, k), k, prefer)


def get_sp_evidence_for_question(question: str, facts: Optional[dict], cfg, k: int = 8) -> List[Evidence]:
    """Evidence pack for general Q&A over SharePoint store."""
    try:
        vs = init_sharepoint_store(cfg)
    except Exception:
        return []
    subqueries = plan_qa_subqueries(question, facts)
    return _gather([], subqueries, question, lambda q, k: _similarity_search(vs, q, k), k, prefer_types=None)


def get_store_evidence(store, question: str, facts: Optional[dict], k: int = 8) -> List[Evidence]:
    """Evidence pack for Q&A against any given vector store (e.g., uploaded RFP store)."""
    if store is None:
        return []
    subqueries = plan_qa_subqueries(question, facts)
    return _gather([], subqueries, question, lambda q, k: _similarity_search(store, q, k), k, prefer_types=None)
