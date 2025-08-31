# modules/proposal_builder.py
# -----------------------------------------------------------------------------
# Build the final proposal document:
# - Merge a DOCX template with static sections and dynamic (AI) sections
# - Honor anchors (content controls/bookmarks/marker text)
# - Insert headings correctly (style-aware) and preserve template formatting
# - Optional TOC, page breaks, and sources line
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from io import BytesIO

from docx import Document
from docx.document import Document as _Doc
from docx.text.paragraph import Paragraph
from docx.oxml.shared import OxmlElement, qn

from modules.docx_generator import (
    append_docx,
    insert_markdown_block,
    add_page_break,
    add_heading_paragraph,
    find_anchor_paragraph,
    add_sources_line,
    normalize_static_section_heading,
    insert_paragraph_after,   # NEW: OXML-based insert
)
from modules.sp_retrieval import get_sp_evidence
from modules.utils import safe_read_docx


# ---------------------------- Options & types ----------------------------

@dataclass
class BuildOptions:
    use_anchors: bool = True
    template_has_headings: bool = True
    page_breaks: bool = True
    include_sources: bool = True
    add_toc: bool = True
    rec_style: str = "bullets"             # "bullets" | "paragraphs"
    top_k_default: int = 6
    top_k_per_section: int = 0             # 0 → use default
    tone: str = "Professional"
    static_heading_mode: str = "demote"    # 'keep' | 'demote' | 'strip'
    facts: dict = field(default_factory=dict)

    # Dynamic heading behavior
    force_dynamic_heading: bool = True # force insert a heading for dynamic sections regardless
    dynamic_heading_level: int = 1         # 1..6
    dynamic_heading_style_name: Optional[str] = None  # e.g., "Heading 2"


# ---------------------------- Utilities ----------------------------

_HEADING_STYLE_PREFIX = "Heading "

def _best_heading_style(doc: _Doc, level: int, override: Optional[str]) -> str:
    """Return a valid heading style present in doc."""
    if override and override in doc.styles:
        return override
    candidate = f"{_HEADING_STYLE_PREFIX}{max(1, min(6, level))}"
    if candidate in doc.styles:
        return candidate
    # Fallbacks
    for name in [candidate, "Heading1", "Heading 1", "Title"]:
        if name in doc.styles:
            return name
    for st in doc.styles:
        if str(st.name).lower().startswith("heading"):
            return st.name
    return "Normal"


def _is_heading(p: Paragraph) -> bool:
    try:
        sname = (p.style.name if p.style else "") or ""
        return sname.lower().startswith("heading")
    except Exception:
        return False


def _ensure_heading_at_anchor(base_doc: _Doc, label: str, opts: BuildOptions) -> Paragraph:
    """
    Ensure there's a visible heading paragraph for this dynamic label.
    If the anchor paragraph isn't a heading, we insert a heading just *after* the anchor.
    Returns the heading paragraph to which content should be appended.
    """
    anchor_p = find_anchor_paragraph(base_doc, label)
    if anchor_p is None:
        # No anchor → append at end, create heading there
        if opts.page_breaks:
            add_page_break(base_doc)
        hstyle = _best_heading_style(base_doc, opts.dynamic_heading_level, opts.dynamic_heading_style_name)
        return add_heading_paragraph(base_doc, label, style_name=hstyle)

    # Anchor found
    if _is_heading(anchor_p):
        return anchor_p

    # Insert a proper heading right after the anchor
    hstyle = _best_heading_style(base_doc, opts.dynamic_heading_level, opts.dynamic_heading_style_name)
    new_p = insert_paragraph_after(anchor_p, text=label, style_name=hstyle)
    return new_p


def _target_for_dynamic(base_doc: _Doc, label: str, opts: BuildOptions) -> Paragraph:
    """
    Decide the paragraph under which dynamic content should be inserted.
    We always end up with a heading paragraph returned here.
    """
    if not opts.use_anchors:
        if opts.page_breaks:
            add_page_break(base_doc)
        hstyle = _best_heading_style(base_doc, opts.dynamic_heading_level, opts.dynamic_heading_style_name)
        return add_heading_paragraph(base_doc, label, style_name=hstyle)

    heading_p = _ensure_heading_at_anchor(base_doc, label, opts)

    # If forced heading and the anchor-heading text is different, insert our own heading below
    if opts.force_dynamic_heading:
        txt = (heading_p.text or "").strip()
        if txt.lower() != label.lower():
            hstyle = _best_heading_style(base_doc, opts.dynamic_heading_level, opts.dynamic_heading_style_name)
            heading_p = insert_paragraph_after(heading_p, text=label, style_name=hstyle)

    return heading_p


def _insert_static_section(base_doc: _Doc, label: str, path: str, opts: BuildOptions) -> None:
    """
    Insert a *static* section from a mapped DOCX file.
    We optionally normalize the first heading (demote/strip/keep) to avoid duplicate headings vs. template.
    """
    sec_doc = safe_read_docx(path)
    normalize_static_section_heading(sec_doc, mode=opts.static_heading_mode)

    if opts.use_anchors:
        anchor_p = find_anchor_paragraph(base_doc, label)
        if anchor_p is not None:
            idx = base_doc.paragraphs.index(anchor_p)
            append_docx(base_doc, sec_doc, after_index=idx)
            return

    # default: end of document
    if opts.page_breaks:
        add_page_break(base_doc)
    append_docx(base_doc, sec_doc, after_index=None)


def _insert_dynamic_section(base_doc: _Doc, label: str, md_text: str, evidence: List[dict], opts: BuildOptions) -> None:
    """
    Insert a *dynamic* section:
    - Ensure a proper heading (style-aware)
    - Insert markdown content (bullets/paragraphs) *immediately after* heading
    - Optionally add a 'Sources:' line right after the content block
    """
    heading_p = _target_for_dynamic(base_doc, label, opts)

    # Insert content after heading (returns last inserted paragraph)
    last_para = insert_markdown_block(base_doc, md_text, after_paragraph=heading_p)

    # Sources line (distinct paragraph right after content)
    if opts.include_sources and evidence:
        sources = []
        seen = set()
        for ev in evidence:
            src = ev.get("source") or ""
            if src and src not in seen:
                seen.add(src); sources.append(src)
        if sources:
            add_sources_line(base_doc, sources, after_paragraph=last_para)


def _insert_toc(base_doc: _Doc, title: str = "Table of Contents") -> None:
    """Minimal TOC (Word updates fields when opened)."""
    hstyle = _best_heading_style(base_doc, 1, None)
    add_heading_paragraph(base_doc, title, style_name=hstyle)
    p = base_doc.add_paragraph()
    fld = OxmlElement('w:fldSimple')
    fld.set(qn('w:instr'), 'TOC \\o "1-3" \\h \\z \\u')
    p._p.append(fld)


# ---------------------------- Public API ----------------------------

def build_proposal(
    template_path: str,
    static_paths: Dict[str, str],
    final_order: List[str],
    oai,
    cfg,
    opts: BuildOptions,
):
    """
    Build the final proposal DOCX (single file) + optional 'recommendations' DOCX.
    Returns:
        (draft_docx_bytes, recos_docx_bytes, preview_dict, meta_dict)
    preview_dict: {section_label: {"md": "...", "sources": [...], "evidence":[...]}}
    """
    base_doc = Document(template_path)

    if opts.add_toc:
        _insert_toc(base_doc, title="Table of Contents")

    preview: Dict[str, dict] = {}
    for item in final_order:
        if item.startswith("[Dyn] "):
            label = item[6:].strip()
            # retrieve SharePoint evidence
            k = opts.top_k_per_section or opts.top_k_default
            ev = get_sp_evidence(label, opts.facts, cfg, k=k)

            # compose section text
            if not ev:
                md = f"*No high-confidence SharePoint evidence found for **{label}**. Consider re-indexing or adjusting your query.*"
                evidence = []
            else:
                numbered = []
                uniq_sources = []
                for i, e in enumerate(ev, start=1):
                    src = e.source + (f" p.{e.page_hint}" if e.page_hint else "")
                    if e.source and e.source not in uniq_sources:
                        uniq_sources.append(e.source)
                    why = f" — {e.why_relevant}" if e.why_relevant else ""
                    numbered.append(f"[{i}] {e.text}\n(Source: {src}{why})")
                context = "\n\n".join(numbered)[:8000]
                style_hint = "bullet points" if opts.rec_style == "bullets" else "tight paragraphs"
                prompt = (
                    f"Write the section **{label}** for a proposal.\n"
                    f"- style: {style_hint}, tone: {opts.tone}\n"
                    "- base it strictly on the numbered evidence; use no external facts.\n"
                    "- append [^n] to claims grounded in evidence [n].\n\n"
                    f"EVIDENCE:\n{context}\n"
                )
                try:
                    res = oai.chat.completions.create(
                        model=cfg.ANALYSIS_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                    )
                    md = res.choices[0].message.content.strip()
                except Exception as ex:
                    md = f"*LLM error while composing **{label}***: {ex}"
                evidence = [vars(x) for x in ev]

            # insert into the document with proper heading & placement
            _insert_dynamic_section(base_doc, label, md, evidence, opts)
            preview[label] = {"md": md, "sources": list({e.get('source','') for e in evidence if e.get('source')}), "evidence": evidence}

        else:
            label = item.strip()
            path = static_paths.get(label)
            if not path:
                continue
            _insert_static_section(base_doc, label, path, opts)

    # finalize
    out = BytesIO()
    base_doc.save(out)
    draft_bytes = out.getvalue()

    # optional recommendations doc
    recs_bytes = None
    if preview:
        doc2 = Document()
        add_heading_paragraph(doc2, "Dynamic Recommendations (Preview)", style_name=_best_heading_style(doc2, 1, None))
        for sec, pkg in preview.items():
            add_heading_paragraph(doc2, sec, style_name=_best_heading_style(doc2, 2, None))
            last_p = insert_markdown_block(doc2, pkg["md"])
            if opts.include_sources and pkg.get("sources"):
                add_sources_line(doc2, pkg["sources"], after_paragraph=last_p)
        b2 = BytesIO()
        doc2.save(b2)
        recs_bytes = b2.getvalue()

    meta = {"template": template_path, "final_order": final_order}
    return draft_bytes, recs_bytes, preview, meta
