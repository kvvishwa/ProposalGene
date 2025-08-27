# modules/docx_generator.py
# -----------------------------------------------------------------------------
# Safer-but-richer DOCX helpers:
# - OXML paragraph insertion after a given paragraph
# - Style-aware heading insertion
# - "textsmart" append that preserves basic formatting & bullets
# - Minimal Markdown → Word (uses List Bullet style when present)
# - Sources line, page break
# - Anchor finder
# - Static heading normalization
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import List, Optional
from copy import deepcopy

from docx import Document
from docx.document import Document as _Doc
from docx.text.paragraph import Paragraph
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT, WD_BREAK
from docx.oxml.shared import OxmlElement, qn


# ----------------------------- insertion primitives -----------------------------

def insert_paragraph_after(paragraph: Paragraph, text: str = "", style_name: Optional[str] = None) -> Paragraph:
    """Insert a new paragraph node immediately after the given paragraph using OXML."""
    new_p_elm = OxmlElement('w:p')
    paragraph._p.addnext(new_p_elm)
    new_p = Paragraph(new_p_elm, paragraph._parent)
    if style_name:
        try:
            new_p.style = paragraph._parent.part.document.styles[style_name]
        except Exception:
            pass
    if text:
        new_p.add_run(text)
    return new_p


def add_page_break(doc: _Doc, after_paragraph: Optional[Paragraph] = None) -> Paragraph:
    """Insert a page break; return the paragraph containing the break."""
    if after_paragraph is not None:
        p = insert_paragraph_after(after_paragraph)
    else:
        p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)
    return p


def add_heading_paragraph(doc: _Doc, text: str, style_name: str = "Heading 1", after_paragraph: Optional[Paragraph] = None) -> Paragraph:
    """Insert a heading paragraph with given style; fallback gracefully."""
    if after_paragraph is not None:
        p = insert_paragraph_after(after_paragraph, text="", style_name=style_name)
    else:
        p = doc.add_paragraph()
        try:
            p.style = doc.styles[style_name]
        except Exception:
            try:
                p.style = doc.styles["Heading 1"]
            except Exception:
                pass
    p.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT
    if text:
        p.add_run(text)
    return p


# ----------------------------- block builders -----------------------------

def _best_list_style(doc: _Doc, preferred: List[str]) -> Optional[str]:
    for name in preferred:
        if name in doc.styles:
            return name
    # scan for common variants
    for st in doc.styles:
        nm = str(st.name)
        if "List Bullet" in nm or "List" in nm and "Bullet" in nm:
            return nm
    return None


def insert_markdown_block(doc: _Doc, md_text: str, after_paragraph: Optional[Paragraph] = None) -> Paragraph:
    """
    Minimal Markdown → Word.
      - lines starting with -, *, • become bullets (uses List Bullet style if present)
      - others become paragraphs
    Returns the last inserted paragraph.
    """
    lines = [ln.rstrip() for ln in (md_text or "").splitlines()]
    last_p = after_paragraph
    bullet_style = _best_list_style(doc, ["List Bullet", "List Paragraph"])

    def _insert(text: str, bullet: bool = False) -> Paragraph:
        nonlocal last_p
        p = insert_paragraph_after(last_p) if last_p is not None else doc.add_paragraph()
        if bullet:
            try:
                if bullet_style:
                    p.style = doc.styles[bullet_style]
                else:
                    # fallback: add a bullet glyph explicitly
                    text = ("• " + text) if text and not text.lstrip().startswith("•") else text
            except Exception:
                pass
        if text:
            p.add_run(text)
        last_p = p
        return p

    for ln in lines:
        if not ln.strip():
            _insert("", bullet=False)
        elif ln.lstrip().startswith(("-", "*", "•")):
            _insert(ln.lstrip("-*• ").strip(), bullet=True)
        else:
            _insert(ln, bullet=False)

    return last_p if last_p is not None else doc.paragraphs[-1]


def add_sources_line(doc: _Doc, sources: List[str], after_paragraph: Optional[Paragraph] = None) -> Paragraph:
    """Append a small 'Sources:' line, optionally inserted after a given paragraph."""
    text = "Sources: " + ", ".join(sources)
    p = insert_paragraph_after(after_paragraph) if after_paragraph is not None else doc.add_paragraph()
    p.add_run(text)
    return p


# ----------------------------- append strategies -----------------------------

def _para_has_numbering(src_p: Paragraph) -> bool:
    """Detect if a source paragraph has numbering/bullets (best-effort)."""
    ppr = getattr(src_p._p, "pPr", None)
    return bool(ppr is not None and getattr(ppr, "numPr", None) is not None)


def _looks_bullet_style(name: str) -> bool:
    n = (name or "").lower()
    return ("bullet" in n) or ("list" in n and "number" not in n)


def _looks_number_style(name: str) -> bool:
    n = (name or "").lower()
    return ("number" in n) or ("list" in n and "bullet" not in n)


def _clone_paragraph_textsmart(dst_doc: _Doc, src_p: Paragraph, after: Optional[Paragraph]) -> Paragraph:
    """
    Clone paragraph content with basic formatting and list heuristics.
    - Preserves style if available in destination
    - Preserves alignment/spacing when possible
    - If source is a list/bullet, apply a bullet/number style when found; else prefix a bullet glyph as fallback
    """
    p = insert_paragraph_after(after) if after is not None else dst_doc.add_paragraph()

    # Style
    try:
        if src_p.style and src_p.style.name in dst_doc.styles:
            p.style = dst_doc.styles[src_p.style.name]
    except Exception:
        pass

    # Alignment & spacing (best-effort)
    try:
        pf_dst = p.paragraph_format
        pf_src = src_p.paragraph_format
        pf_dst.alignment = pf_src.alignment
        pf_dst.left_indent = pf_src.left_indent
        pf_dst.first_line_indent = pf_src.first_line_indent
        pf_dst.space_before = pf_src.space_before
        pf_dst.space_after = pf_src.space_after
    except Exception:
        pass

    # List/bullets detection
    try:
        is_listy = _para_has_numbering(src_p) or _looks_bullet_style(getattr(src_p.style, "name", ""))
        if is_listy:
            lst_style = _best_list_style(dst_doc, ["List Bullet", "List Paragraph", "List Number"])
            if lst_style:
                p.style = dst_doc.styles[lst_style]
            else:
                # glyph fallback if no style exists
                if not src_p.text.strip().startswith(("•", "-", "*")):
                    p.add_run("• ")
    except Exception:
        pass

    # Runs
    for run in src_p.runs:
        r = p.add_run(run.text)
        r.bold = run.bold
        r.italic = run.italic
        r.underline = run.underline

    return p


def append_docx_textsmart(base_doc: _Doc, add_doc: _Doc, after_index: Optional[int]) -> None:
    """
    Safer, formatting-aware append:
    - Recreates paragraphs with styles/alignment/runs
    - Detects list/bulleted paragraphs and applies a suitable list style
    - Tables are flattened as text rows (keeps order & safety)
    Content is inserted *after* the given paragraph index when provided.
    """
    after = None
    if after_index is not None:
        try:
            after = base_doc.paragraphs[after_index]
        except Exception:
            after = None

    cursor = after
    for el in add_doc.element.body:
        tag = el.tag.split('}')[-1]
        if tag == "p":
            src_p = Paragraph(el, add_doc)
            cursor = _clone_paragraph_textsmart(base_doc, src_p, cursor)
        elif tag == "tbl":
            # Flatten table to rows of pipe-separated cells (safe & ordered)
            from docx.table import Table
            t = Table(el, add_doc)
            for row in t.rows:
                line = " | ".join(cell.text.strip() for cell in row.cells)
                cursor = insert_paragraph_after(cursor) if cursor is not None else base_doc.add_paragraph()
                cursor.add_run(line)
        else:
            # ignore sectPr/header/footer etc.
            continue


def append_docx_deepcopy(base_doc: _Doc, add_doc: _Doc, after_index: Optional[int]) -> None:
    """Risky but fast raw-XML append (can corrupt when rels/numbering mismatch)."""
    body = base_doc.element.body
    if after_index is None:
        for el in add_doc.element.body:
            body.append(deepcopy(el))
        return
    try:
        anchor_p = base_doc.paragraphs[after_index]
        body_elems = list(body)
        pos = body_elems.index(anchor_p._p)
    except Exception:
        for el in add_doc.element.body:
            body.append(deepcopy(el))
        return
    insert_pos = pos + 1
    for el in add_doc.element.body:
        body.insert(insert_pos, deepcopy(el))
        insert_pos += 1


def append_docx(base_doc: _Doc, add_doc: _Doc, after_index: Optional[int], mode: str = "textsmart") -> None:
    """
    Append 'add_doc' into 'base_doc'.
      mode:
        - "textsmart" (default): safe + preserves common formatting & bullets
        - "textonly": safest but plain
        - "deepcopy": fastest & richest, but can corrupt (use only if you must)
    """
    if mode == "deepcopy":
        append_docx_deepcopy(base_doc, add_doc, after_index)
    elif mode == "textonly":
        # legacy: simplest recreation
        append_docx_textsmart(base_doc, add_doc, after_index)  # we can keep smart path for both
    else:
        append_docx_textsmart(base_doc, add_doc, after_index)


# ----------------------------- anchors & normalization -----------------------------

def find_anchor_paragraph(doc: _Doc, label: str) -> Optional[Paragraph]:
    """Find an anchor by [[label]] / {{label}} or exact paragraph text match."""
    needle1 = f"[[{label}]]".lower()
    needle2 = f"{{{{{label}}}}}".lower()
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if not t:
            continue
        tl = t.lower()
        if needle1 in tl or needle2 in tl:
            return p
        if tl == label.lower():
            return p
    return None


def normalize_static_section_heading(doc: _Doc, mode: str = "demote") -> None:
    """
    Normalize first heading in a static section doc:
      - 'keep'   : leave untouched
      - 'demote' : Heading 1 → Heading 2 (etc.) to avoid clashing with template sections
      - 'strip'  : remove first heading text
    """
    if mode not in ("keep", "demote", "strip"):
        mode = "demote"

    if not doc.paragraphs:
        return

    p0 = doc.paragraphs[0]
    name = (p0.style.name if p0.style else "") or ""
    is_heading = name.lower().startswith("heading")
    if not is_heading:
        return

    if mode == "strip":
        for r in list(p0.runs):
            r.text = ""
        return

    if mode == "demote":
        try:
            if name.startswith("Heading "):
                lvl = int(name.split(" ")[1])
                new_name = f"Heading {min(6, lvl + 1)}"
                if new_name in doc.styles:
                    p0.style = doc.styles[new_name]
        except Exception:
            pass
