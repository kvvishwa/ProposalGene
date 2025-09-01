# modules/sp_retrieval.py
# -----------------------------------------------------------------------------
# Evidence retrieval helpers for SharePoint and Uploaded stores.
# Cascading retrieval (MMR -> similarity -> widen -> keyword fallback),
# tolerant scoring and per-source caps to avoid "too strict" empty results.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ----------------------------- Data Model ------------------------------------

@dataclass
class Evidence:
    text: str
    source: str
    page_hint: Optional[str] = None
    why_relevant: Optional[str] = None


# ----------------------------- Utilities -------------------------------------

def _as_text(doc: Any) -> str:
    return (
        getattr(doc, "page_content", None)
        or getattr(doc, "content", None)
        or (str(doc) if doc is not None else "")
    ) or ""


def _as_meta(doc: Any) -> Dict[str, Any]:
    meta = getattr(doc, "metadata", None)
    if isinstance(meta, dict):
        return meta
    return {}


def _mk_ev(doc: Any) -> Evidence:
    txt = (_as_text(doc) or "").strip()
    meta = _as_meta(doc)
    src = str(meta.get("source", "") or meta.get("file", "") or "")
    page = meta.get("page") or meta.get("page_number") or meta.get("pageno")
    page_hint = f"p.{page}" if page else None
    return Evidence(text=txt, source=src, page_hint=page_hint, why_relevant=None)


def _tokenize(s: str) -> List[str]:
    return [t for t in (s or "").lower().replace("/", " ").replace("-", " ").split() if t]


def _uniq(seq: Iterable[str]) -> List[str]:
    out, seen = [], set()
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out


def _soft_overlap_score(query: str, text: str) -> float:
    """Lightweight lexical overlap score in [0,1]."""
    q = set(_tokenize(query))
    if not q:
        return 0.0
    t = set(_tokenize(text))
    inter = len(q & t)
    return inter / max(1.0, len(q))


def _score_snippet(query: str, text: str, meta: Dict[str, Any]) -> float:
    """
    Combine lexical overlap + gentle length shaping.
    Keep this forgiving: we don't want to zero-out plausible hits.
    """
    base = _soft_overlap_score(query, text)
    L = len(text)
    # Gentle preference for mid-sized chunks
    if L < 250:
        base *= 0.9
    elif L > 1800:
        base *= 0.95
    # Mild boost if page info present (often indicates PDF page chunking)
    if "page" in meta or "page_number" in meta or "pageno" in meta:
        base += 0.03
    return float(max(0.0, min(1.0, base + 0.001)))


def _dedup_key(ev: Evidence) -> str:
    return (ev.source + "|" + (ev.text[:200] if ev.text else "")).lower()


def _limit_per_source(evs: Sequence[Evidence], k: int, per_source: int = 3) -> List[Evidence]:
    out: List[Evidence] = []
    per: Dict[str, int] = {}
    for e in evs:
        cnt = per.get(e.source or "", 0)
        if cnt < per_source:
            out.append(e)
            per[e.source or ""] = cnt + 1
        if len(out) >= k:
            break
    return out


# -------------------------- Store Adapters -----------------------------------

def _mmr_pass(store: Any, query: str, k: int) -> List[Evidence]:
    """Try MMR for diversity; fall back gracefully."""
    retr = None
    try:
        retr = store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": max(25, k * 5),
                "lambda_mult": 0.3,
            },
        )
    except Exception:
        retr = None
    if retr:
        try:
            docs = retr.get_relevant_documents(query)
            return [_mk_ev(d) for d in docs or []]
        except Exception:
            return []
    return []


def _sim_with_scores(store: Any, query: str, k: int) -> Tuple[List[Evidence], List[float]]:
    """
    Try similarity_search_with_score if available, else similarity_search (fake scores).
    Some stores return distances (lower=better). We normalize to similarity (higher=better).
    """
    func = getattr(store, "similarity_search_with_score", None)
    if func:
        try:
            pairs = func(query, k=k) or []
            evs: List[Evidence] = []
            sims: List[float] = []
            for d, s in pairs:
                evs.append(_mk_ev(d))
                try:
                    s = float(s)
                except Exception:
                    s = 0.0
                # Convert distance→similarity if needed (heuristic)
                sims.append(1.0 - s if s > 1.0 else s)
            return evs, sims
        except Exception:
            pass
    try:
        docs = store.similarity_search(query, k=k) or []
        return [_mk_ev(d) for d in docs], [1.0] * len(docs)
    except Exception:
        return [], []


def _keyword_fallback(store: Any, query: str, limit: int) -> List[Evidence]:
    """
    If dense retrieval returns nothing, try simple keyword match.
    Works with our FAISS adapter exposing get_all_texts(); otherwise no-op.
    """
    get_all = getattr(store, "get_all_texts", None)
    if not callable(get_all):
        return []
    toks = [t for t in _tokenize(query) if len(t) >= 3]
    if not toks:
        return []
    seen: set[str] = set()
    out: List[Evidence] = []
    for text, meta in get_all() or []:
        low = (text or "").lower()
        if any(t in low for t in toks):
            src = str((meta or {}).get("source", ""))
            ev = Evidence(text=(text or "")[:1600], source=src, page_hint=None, why_relevant=None)
            sig = _dedup_key(ev)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(ev)
            if len(out) >= limit:
                break
    return out


def _cascade_search(store: Any, query: str, k: int) -> List[Evidence]:
    """
    Multi-pass retrieval:
      1) MMR (diverse)
      2) Similarity top-k
      3) Widened similarity if top looks weak
      4) Keyword fallback
    """
    # 1) MMR
    ev_mmr = _mmr_pass(store, query, k=max(6, k))
    if ev_mmr:
        base = ev_mmr
    else:
        base = []

    # 2) Similarity
    ev_sim, scores = _sim_with_scores(store, query, k=max(10, k))
    base.extend(ev_sim)

    # If similarity set seems thin, widen
    def _weak(scores_: List[float]) -> bool:
        return not scores_ or (scores_[0] < 0.2)

    if _weak(scores):
        ev_wide, _ = _sim_with_scores(store, query, k=max(24, k * 4))
        base.extend(ev_wide)

    # 3) Dedup
    deduped: List[Evidence] = []
    seen: set[str] = set()
    for e in base:
        if not e or not (e.text or "").strip():
            continue
        sig = _dedup_key(e)
        if sig in seen:
            continue
        seen.add(sig)
        # trim long text for UI
        e.text = (e.text or "")[:1600]
        deduped.append(e)

    # 4) If still empty, keyword fallback
    if not deduped:
        deduped = _keyword_fallback(store, query, limit=max(10, k)) or []

    # Score & rank
    ranked = sorted(
        deduped,
        key=lambda ev: _score_snippet(query, ev.text or "", {"page_hint": ev.page_hint}),
        reverse=True,
    )

    # Cap per source, total k
    return _limit_per_source(ranked, k=k, per_source=3)


# -------------------------- Query Expansion ----------------------------------

def _extract_terms_from_facts(rfp_facts: Optional[dict]) -> List[str]:
    """
    Pull a few useful terms from facts to enrich the query slightly.
    We keep this conservative to avoid over-filtering.
    """
    if not isinstance(rfp_facts, dict):
        return []
    keys = [
        "solicitation", "title", "project_name", "scope", "services", "department",
        "agency", "contract_and_compliance", "evaluation_and_selection"
    ]
    terms: List[str] = []
    for k in keys:
        v = rfp_facts.get(k)
        if isinstance(v, str) and v.strip():
            terms.append(v.strip())
        elif isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, str) and vv.strip():
                    terms.append(vv.strip())
        elif isinstance(v, list):
            for vv in v:
                if isinstance(vv, str) and vv.strip():
                    terms.append(vv.strip())
    # keep only a handful to avoid overspecializing
    flat = " ".join(terms)
    toks = _tokenize(flat)
    # prefer longer tokens
    toks = [t for t in toks if len(t) >= 5]
    return _uniq(toks)[:8]


def _expand_queries(question: str, rfp_facts: Optional[dict]) -> List[str]:
    # Base question plus 1–2 slight enrichments
    out = [question.strip()]
    extra = _extract_terms_from_facts(rfp_facts)
    if extra:
        out.append(f"{question} " + " ".join(extra[:4]))
    return _uniq([q for q in out if q])


# ------------------------------ Public API -----------------------------------

def get_store_evidence(store: Any, question: str, rfp_facts: Optional[dict] = None, k: int = 6) -> List[Evidence]:
    """
    Evidence from the uploaded-docs store (Chroma in-memory or FAISS).
    """
    subqs = _expand_queries(question, rfp_facts)
    pool: List[Evidence] = []
    seen: set[str] = set()

    for q in subqs:
        evs = _cascade_search(store, q, k=max(10, k))
        for ev in evs:
            sig = _dedup_key(ev)
            if sig in seen:
                continue
            seen.add(sig)
            pool.append(ev)

    # Re-rank on the original question to avoid bias from expansions
    ranked = sorted(
        pool,
        key=lambda ev: _score_snippet(question, ev.text or "", {"page_hint": ev.page_hint}),
        reverse=True,
    )

    final = _limit_per_source(ranked, k=k, per_source=3)
    for e in final:
        if not e.why_relevant:
            e.why_relevant = "Relevant to your question by semantic/keyword match."
    return final


def get_sp_evidence_for_question(question: str, rfp_facts: Optional[dict], cfg, k: int = 6) -> List[Evidence]:
    """
    Evidence from the SharePoint store (persistent Chroma when possible; FAISS fallback).
    """
    # Local import to avoid cycles
    try:
        from modules.vectorstore import init_sharepoint_store
    except ModuleNotFoundError:
        from vectorstore import init_sharepoint_store  # type: ignore

    store = init_sharepoint_store(cfg)

    subqs = _expand_queries(question, rfp_facts)
    pool: List[Evidence] = []
    seen: set[str] = set()

    for q in subqs:
        evs = _cascade_search(store, q, k=max(10, k))
        for ev in evs:
            sig = _dedup_key(ev)
            if sig in seen:
                continue
            seen.add(sig)
            pool.append(ev)

    ranked = sorted(
        pool,
        key=lambda ev: _score_snippet(question, ev.text or "", {"page_hint": ev.page_hint}),
        reverse=True,
    )
    final = _limit_per_source(ranked, k=k, per_source=3)
    for e in final:
        if not e.why_relevant:
            e.why_relevant = "Matches policy/topic terms in your question."
    return final
