# vectorstore.py

import os
from pathlib import Path
from typing import List, Optional, Iterable

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys
import pysqlite3  # comes from pysqlite3-binary
import sqlite3 as _sqlite3
# --- SQLite shim for Azure App Service (Chroma needs sqlite >= 3.35) ---
# If the system sqlite3 is old, force Python to use the one bundled with pysqlite3-binary.
try:
    
    ver_tuple = tuple(int(x) for x in _sqlite3.sqlite_version.split("."))
    _NEEDS_SHIM = ver_tuple < (3, 35, 0)
except Exception:
    _NEEDS_SHIM = True

if _NEEDS_SHIM:
    
    sys.modules["sqlite3"] = pysqlite3
    sys.modules["sqlite3.dbapi2"] = pysqlite3.dbapi2
# --- end shim ---

# --- robust local imports ---
try:
    from text_extraction import extract_text
except ModuleNotFoundError:
    from modules.text_extraction import extract_text  # fallback

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
    return HuggingFaceEmbeddings(model_name=EMB_MODEL)

def _text_splitter(cfg):
    return RecursiveCharacterTextSplitter(
        chunk_size=getattr(cfg, "CHUNK_SIZE", 1000),
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
    return Chroma(collection_name="uploaded_docs", persist_directory=str(p), embedding_function=_embeddings())

def init_sharepoint_store(cfg) -> Optional[Chroma]:
    p = _vec_dir_for(cfg, "sharepoint")
    return Chroma(collection_name="sharepoint_docs", persist_directory=str(p), embedding_function=_embeddings())

def wipe_up_store(cfg):
    p = _vec_dir_for(cfg, "uploaded")
    if p.exists():
        for f in p.glob("**/*"):
            try: f.unlink()
            except: pass
        try: p.rmdir()
        except: pass

def wipe_sp_store(cfg):
    p = _vec_dir_for(cfg, "sharepoint")
    if p.exists():
        for f in p.glob("**/*"):
            try: f.unlink()
            except: pass
        try: p.rmdir()
        except: pass

def ingest_files(paths: Iterable[str], store: Chroma, chunk_size: int = 1000) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
    )
    added = []
    for p in paths:
        try:
            text = extract_text(p)
            if not text:
                continue
            chunks = splitter.split_text(text)
            metadatas = [{"source": p} for _ in chunks]
            store.add_texts(chunks, metadatas=metadatas)
            store.persist()
            added.append(p)
        except Exception:
            continue
    return added

def ingest_sharepoint(cfg, include_exts: Optional[List[str]] = None) -> List[str]:
    """
    Download files from cfg.SP_SITE_URL (via sharepoint.py) and index them into Chroma.
    Tags chunks with sp_category (detect SA-ITPS by path). Returns list of local file paths.
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
            text = extract_text(local_path)
            if not text:
                continue
            chunks = splitter.split_text(text)
            cat = _derive_sp_category(local_path)
            metadatas = [{"source": local_path, "sp_category": cat} for _ in chunks]
            sp_store.add_texts(chunks, metadatas=metadatas)
        except Exception:
            continue

    sp_store.persist()
    return local_files
