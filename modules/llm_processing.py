# modules/llm_processing.py
# -----------------------------------------------------------------------------
# LLM helpers for Understanding and Dynamic Recommendations.
# Now uses SharePoint evidence packs (sp_retrieval) for better grounding & citations.
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any

from modules.sp_retrieval import get_sp_evidence, Evidence  # NEW

# -------------------- Understanding helpers (unchanged) -------------------- #

def generate_proposal_brief(up_store) -> str:
    from modules.app_helpers import rag_answer_uploaded
    answer, _ = rag_answer_uploaded(
        up_store, _get_oai(), _get_cfg(),
        "Provide a 6–10 sentence executive brief of this RFP. Include: issuing entity, purpose, "
        "high-level scope/theme, notable constraints, and due date if present. Use plain prose.",
        top_k=8,
    )
    return answer

def generate_scope_snapshot(up_store) -> str:
    from modules.app_helpers import rag_answer_uploaded
    answer, _ = rag_answer_uploaded(
        up_store, _get_oai(), _get_cfg(),
        "Create a scope snapshot with two headings:\n"
        "IN-SCOPE: bullet list of key deliverables/services; and\n"
        "OUT-OF-SCOPE: bullet list of exclusions/assumptions.\n"
        "Use short bullets. Only use information present in the provided context.",
        top_k=8,
    )
    return answer

def generate_intelligence_questions(up_store, top_k: int = 6) -> List[Dict[str, Any]]:
    from modules.app_helpers import rag_answer_uploaded
    md, _ = rag_answer_uploaded(
        up_store, _get_oai(), _get_cfg(),
        "Given this RFP context, propose 10 clarifying questions we should ask the customer. "
        "For each, give 3–4 multiple-choice options that reflect typical answers. "
        "Return as simple markdown with numbered questions and bullet options.",
        top_k=top_k,
    )
    return _markdown_questions_to_list(md)

def generate_structuring_questions(up_store, rfp_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    from modules.app_helpers import rag_answer_uploaded
    typ = rfp_config.get("type", "Technical Solution")
    depth = rfp_config.get("depth", "Standard")
    tone = rfp_config.get("tone", "Professional")
    audience = rfp_config.get("audience", "Mixed (Business + Technical)")
    extras = ", ".join(rfp_config.get("extras", []))
    prompt = (
        f"You are preparing a proposal for an RFP.\n"
        f"- Type: {typ}\n- Depth: {depth}\n- Tone: {tone}\n- Audience: {audience}\n"
        f"- Preferred sections (seed): {extras or 'none'}\n\n"
        f"From the provided context, propose 8–12 candidate sections we should include. "
        f"Group them into 3 questions:\n"
        f"Q1: Core narrative sections (4–6 options)\n"
        f"Q2: Delivery & management (3–5 options)\n"
        f"Q3: Risk/compliance/assumptions (3–5 options)\n"
        f"Return as markdown with headings and bullet lists of options."
    )
    md, _ = rag_answer_uploaded(up_store, _get_oai(), _get_cfg(), prompt, top_k=8)
    return _markdown_questions_to_list(md)

# ---------------- Dynamic recommendations using evidence (UPDATED) ----------- #

def generate_dynamic_recommendations(
    section_label: str,
    oai,
    cfg,
    *,
    facts: Optional[dict] = None,          # NEW: pass RFP facts to tune retrieval
    style: str = "bullets",
    tone: str = "Professional",
    top_k: int = 6,
    target_words: int = 200,
    max_bullets: int = 6,
    include_sources: bool = True,
) -> Tuple[str, List[str], List[Evidence]]:
    """
    Build SharePoint-grounded recommendations for a single dynamic section using
    the evidence pipeline. Returns (markdown_text, sources_list, evidence_list).
    """
    # 1) retrieve evidence pack
    evidence = get_sp_evidence(section_label, facts, cfg, k=top_k)

    # Build a numbered context block with short “why” annotations
    numbered = []
    uniq_sources: List[str] = []
    for i, ev in enumerate(evidence, start=1):
        src_label = ev.source + (f" {ev.page_hint}" if ev.page_hint else "")
        if ev.source and ev.source not in uniq_sources:
            uniq_sources.append(ev.source)
        why = f" — {ev.why_relevant}" if ev.why_relevant else ""
        numbered.append(f"[{i}] {ev.text}\n(Source: {src_label}{why})")

    context = "\n\n".join(numbered)[:8000]
    has_ctx = bool(context.strip())

    # 2) compose prompt with a simple “citation contract”
    if style not in ("bullets", "paragraphs"):
        style = "bullets"
    style_directive = (
        f"Write {min(max_bullets, 12)} short, evidence-grounded bullets (one idea per bullet)."
        if style == "bullets" else
        f"Write a concise {target_words}–{int(target_words*1.3)} word narrative in 2–4 paragraphs."
    )
    prompt = (
        f"You are drafting the **{section_label}** section for a proposal.\n"
        f"Tone: {tone}. Audience: mixed business + technical evaluators.\n"
        f"{style_directive}\n"
        f"- Cite evidence using footnotes like [^n] where n matches the evidence number below.\n"
        f"- Only include claims supported by the evidence. If no evidence exists, produce a brief generic draft and do not add footnotes.\n"
        f"- Do not repeat the section title in the body (the template provides the heading).\n\n"
        + (f"EVIDENCE:\n{context}\n\n" if has_ctx else "EVIDENCE: (none — produce a short generic best-practice draft)\n\n")
        + f"Now produce the {('bullets' if style=='bullets' else 'section text')}.\n"
    )

    # 3) call the model
    try:
        res = oai.chat.completions.create(
            model=cfg.ANALYSIS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25 if has_ctx else 0.5,
        )
        md = res.choices[0].message.content.strip()
    except Exception as ex:
        md = f"*LLM error while generating '{section_label}': {ex}*"

    # normalize bullets if needed
    if style == "bullets":
        lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
        has_bullets = any(ln.startswith(("-", "*", "•")) for ln in lines)
        if not has_bullets:
            md = "\n".join(f"- {ln}" for ln in lines[:max_bullets])

    # 4) sources line (numbered)
    if include_sources and evidence:
        numbered_sources = []
        seen = {}
        idx = 1
        for ev in evidence:
            if ev.source not in seen:
                label = f"[{idx}] {ev.source}" + (f" {ev.page_hint}" if ev.page_hint else "")
                numbered_sources.append(label)
                seen[ev.source] = idx
                idx += 1
        md += "\n\n_Sources: " + "; ".join(numbered_sources) + "_"

    return md, uniq_sources, evidence

# ---------------- Internal config/oai cache ---------------- #

_cached = {"oai": None, "cfg": None}

def _get_oai():
    if _cached["oai"] is None:
        try:
            from openai import OpenAI
            from config import load_config
            cfg = load_config()
            _cached["oai"] = OpenAI(api_key=cfg.OPENAI_API_KEY)
            _cached["cfg"] = cfg
        except Exception:
            _cached["oai"] = None
    return _cached["oai"]

def _get_cfg():
    if _cached["cfg"] is None:
        try:
            from config import load_config
            _cached["cfg"] = load_config()
        except Exception:
            _cached["cfg"] = None
    return _cached["cfg"]

# ---------------- Minimal markdown → Q list parser --------- #

def _markdown_questions_to_list(md: str) -> List[Dict[str, Any]]:
    if not md:
        return []
    lines = [l.rstrip() for l in md.splitlines()]
    out: List[Dict[str, Any]] = []
    cur_q = None
    for ln in lines:
        if not ln.strip():
            continue
        if any(ln.lstrip().startswith(f"{i}.") for i in range(1, 21)):
            if cur_q:
                out.append(cur_q)
            cur_q = {"question": ln.split(".", 1)[1].strip() if "." in ln else ln.strip(), "options": []}
            continue
        if ln.lstrip().startswith(("-", "*", "•")):
            if cur_q is None:
                cur_q = {"question": "Select one:", "options": []}
            opt = ln.lstrip("-*• ").strip()
            if opt:
                cur_q["options"].append(opt)
    if cur_q:
        out.append(cur_q)
    return out
