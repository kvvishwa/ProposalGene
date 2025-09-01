# modules/vectorstore.py  — Local+ (Chroma-only) with small recall upgrades
import os
from pathlib import Path
from typing import List, Optional, Iterable
import logging

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

log = logging.getLogger(__name__)

# --- robust local imports ---
try:
    from text_extraction import extract_text
except ModuleNotFoundError:
    from modules.text_extraction import extract_text  # fallback

# OPTIONAL: page-aware PDF extraction (adds 'page' metadata if available)
try:
    from modules.text_extraction import pdf_text_with_pages as _pdf_pages  # type: ignore
except Exception:
    _pdf_pages = None

# Try both layouts: root-level and modules/ package
_sp_fetch = _sp_ingest_wrap = None
try:
    # preferred fetcher
    from sharepoint import fetch_files_from_sharepoint as _sp_fetch
    # optional wrapper in your sharepoint.py
    from sharepoint import ingest_sharepoint_files as _sp_ingest_wrap
except ModuleNotFoundError:
    try:
        from modules.sharepoint import fetch_files_from_sharepoint as _sp_fetch
    except Exception:
        _sp_fetch = None
    try:
        from modules.sharepoint import ingest_sharepoint_files as _sp_ingest_wrap
    except Exception:
        _sp_ingest_wrap = None

DEF_BASE = Path(".")
DEF_VEC_DIR = DEF_BASE / "vectorstore"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def _vec_dir_for(cfg, kind: str) -> Path:
    base = Path(getattr(cfg, "VECTORSTORE_DIR", DEF_VEC_DIR))
    p = base / ("uploaded" if kind == "uploaded" else "sharepoint")
    p.mkdir(parents=True, exist_ok=True)
    return p

def _embeddings():
    # UPGRADE 1: normalize embeddings for more consistent cosine scoring
    return HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def _text_splitter(cfg):
    return RecursiveCharacterTextSplitter(
        chunk_size=getattr(cfg, "CHUNK_SIZE", 1000),
        # OPTIONAL: bump to 160–200 for better POC/email recall
        chunk_overlap=getattr(cfg, "CHUNK_OVERLAP", 120),
        separators=["\n\n", "\n", " ", ""]
    )

def _derive_sp_category(path_like: str) -> str:
    rp = (path_like or "").lower()
    if "staff augmentation-it professional services" in rp:
        return "Staff Augmentation-IT Professional Services"
    return "Other"

def init_uploaded_store(cfg) -> Chroma:
    p = _vec_dir_for(cfg, "uploaded")
    store = Chroma(collection_name="uploaded_docs", persist_directory=str(p), embedding_function=_embeddings())
    log.info("[VectorStore] uploaded backend = Chroma (persistent) at %s", str(p))
    return store

def init_sharepoint_store(cfg) -> Chroma:
    p = _vec_dir_for(cfg, "sharepoint")
    store = Chroma(collection_name="sharepoint_docs", persist_directory=str(p), embedding_function=_embeddings())
    log.info("[VectorStore] sharepoint backend = Chroma (persistent) at %s", str(p))
    return store

def wipe_up_store(cfg):
    p = _vec_dir_for(cfg, "uploaded")
    if p.exists():
        for f in p.glob("**/*"):
            try: f.unlink()
            except Exception: pass
        try: p.rmdir()
        except Exception: pass

def wipe_sp_store(cfg):
    p = _vec_dir_for(cfg, "sharepoint")
    if p.exists():
        for f in p.glob("**/*"):
            try: f.unlink()
            except Exception: pass
        try: p.rmdir()
        except Exception: pass

def ingest_files(paths: Iterable[str], store: Chroma, chunk_size: int = 1000) -> List[str]:
    """
    Ingest uploaded files into the provided Chroma store.
    UPGRADE 2: page-aware PDF indexing (if pdf_text_with_pages is available)
    UPGRADE 3: warn when no text is extracted (scanned PDFs)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=120,            # keep your default; tune if needed
        separators=["\n\n", "\n", " ", ""]
    )
    added: List[str] = []
    for p in paths:
        try:
            ext = Path(p).suffix.lower()
            if ext == ".pdf" and _pdf_pages:
                pages = _pdf_pages(p) or []
                if not pages:
                    log.warning("[Ingest] No text extracted (PDF): %s — scanned PDF? (OCR deps missing)", p)
                    continue
                for txt, pg in pages:
                    if not (txt or "").strip():
                        continue
                    chunks = splitter.split_text(txt)
                    metadatas = [{"source": p, "page": pg} for _ in chunks]
                    store.add_texts(chunks, metadatas=metadatas)
            else:
                text = extract_text(p)
                if not text or not text.strip():
                    log.warning("[Ingest] No text extracted: %s", p)
                    continue
                chunks = splitter.split_text(text)
                metadatas = [{"source": p} for _ in chunks]
                store.add_texts(chunks, metadatas=metadatas)
            store.persist()
            added.append(p)
        except Exception as ex:
            log.warning("[Ingest] Skipped %s: %s", p, ex)
            continue
    return added

def ingest_sharepoint(cfg, include_exts: Optional[List[str]] = None) -> List[str]:
    """
    Download files from cfg.SP_SITE_URL (via sharepoint.py) and index them into Chroma.
    Tags chunks with sp_category (detect SA-ITPS by path). Returns list of local file paths.
    UPGRADE 2: page-aware PDF indexing (if pdf_text_with_pages is available)
    UPGRADE 3: warn when no text is extracted
    """
    if include_exts is None:
        include_exts = ["pdf", "docx", "pptx"]

    if _sp_fetch is None and _sp_ingest_wrap is None:
        raise RuntimeError(
            "SharePoint fetcher not available. Ensure sharepoint.py is importable "
            "and exposes fetch_files_from_sharepoint or ingest_sharepoint_files."
        )

    if _sp_fetch is not None:
        local_files = _sp_fetch(cfg, include_exts=include_exts) or []
    else:
        local_files = _sp_ingest_wrap(cfg, include_exts=include_exts) or []

    if not local_files:
        return []

    sp_store = init_sharepoint_store(cfg)
    splitter = _text_splitter(cfg)

    for local_path in local_files:
        try:
            ext = Path(local_path).suffix.lower()
            cat = _derive_sp_category(local_path)
            if ext == ".pdf" and _pdf_pages:
                pages = _pdf_pages(local_path) or []
                if not pages:
                    log.warning("[SP Ingest] No text (PDF): %s", local_path)
                    continue
                for txt, pg in pages:
                    if not (txt or "").strip():
                        continue
                    chunks = splitter.split_text(txt)
                    metadatas = [{"source": local_path, "sp_category": cat, "page": pg} for _ in chunks]
                    sp_store.add_texts(chunks, metadatas=metadatas)
            else:
                text = extract_text(local_path)
                if not text or not text.strip():
                    log.warning("[SP Ingest] No text: %s", local_path)
                    continue
                chunks = splitter.split_text(text)
                metadatas = [{"source": local_path, "sp_category": cat} for _ in chunks]
                sp_store.add_texts(chunks, metadatas=metadatas)
        except Exception as ex:
            log.warning("[SP Ingest] Skipped %s: %s", local_path, ex)
            continue

    sp_store.persist()
    return local_files
