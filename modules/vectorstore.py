# modules/vectorstore.py
from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- robust local imports (support running as a package or flat files) ---
try:
    from text_extraction import extract_text
except ModuleNotFoundError:
    from modules.text_extraction import extract_text  # fallback

_sp_fetch = _sp_ingest_wrap = None
try:
    # preferred names as in your repo
    from sharepoint import fetch_files_from_sharepoint as _sp_fetch
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

# Try to import Chroma Settings (varies by chromadb version)
try:
    from chromadb.config import Settings as _ChromaSettings
except Exception:
    _ChromaSettings = None

# -------------------- constants --------------------
DEF_BASE = Path(".")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # same as your previous working setup

# -------------------- path helpers --------------------
def _vec_root(cfg) -> Path:
    """
    Vector root: VECTORSTORE_DIR or BASE_DIR/'vectorstore'.
    This keeps vector data aligned with your other app data on Azure.
    """
    base_dir = Path(getattr(cfg, "BASE_DIR", DEF_BASE))
    vec_dir = getattr(cfg, "VECTORSTORE_DIR", None)
    root = Path(vec_dir) if vec_dir else (base_dir / "vectorstore")
    root.mkdir(parents=True, exist_ok=True)
    return root

def _vec_dir_for(cfg, kind: str) -> Path:
    base = _vec_root(cfg)
    sub = "uploaded" if kind == "uploaded" else "sharepoint"
    p = base / sub
    p.mkdir(parents=True, exist_ok=True)
    return p

# -------------------- backend selection --------------------
def _sqlite_ver_tuple() -> Tuple[int, int, int]:
    try:
        return tuple(int(x) for x in sqlite3.sqlite_version.split("."))
    except Exception:
        return (0, 0, 0)

def _make_client_settings(persist_dir: Path):
    """
    Decide Chroma backend based on platform capabilities.
    - If sqlite3 >= 3.35 -> default sqlite (persistent).
    - Else try duckdb+parquet (persistent).
    - Else in-memory (non-persistent) so the app keeps working.
    Returns: (client_settings_or_None, persist_directory_or_None)
    """
    ver = _sqlite_ver_tuple()

    # Case 1: modern sqlite available -> default sqlite backend
    if ver >= (3, 35, 0):
        if _ChromaSettings:
            return _ChromaSettings(is_persistent=True, persist_directory=str(persist_dir)), str(persist_dir)
        # Older chromadb versions may not require client_settings to persist
        return None, str(persist_dir)

    # Case 2: old sqlite -> try duckdb backend (no sqlite dependency)
    if _ChromaSettings:
        try:
            return _ChromaSettings(
                is_persistent=True,
                persist_directory=str(persist_dir),
                chroma_db_impl="duckdb+parquet",
            ), str(persist_dir)
        except Exception:
            pass

    # Case 3: last resort -> in-memory (no persistence)
    if _ChromaSettings:
        try:
            return _ChromaSettings(is_persistent=False), None
        except Exception:
            pass

    return None, None

# -------------------- embeddings & splitter --------------------
def _embeddings() -> HuggingFaceEmbeddings:
    # You can switch to OpenAIEmbeddings here if you want a lighter Azure build.
    return HuggingFaceEmbeddings(model_name=EMB_MODEL)

def _text_splitter(cfg) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=getattr(cfg, "CHUNK_SIZE", 1000),
        chunk_overlap=getattr(cfg, "CHUNK_OVERLAP", 120),
        separators=["\n\n", "\n", " ", ""],
    )

# -------------------- store factories --------------------
def _new_chroma(collection: str, persist_path: Path, emb) -> Chroma:
    client_settings, persist_dir = _make_client_settings(persist_path)
    # persistent path (sqlite or duckdb)
    if persist_dir:
        return Chroma(
            collection_name=collection,
            persist_directory=persist_dir,
            embedding_function=emb,
            client_settings=client_settings,
        )
    # in-memory fallback (no persist directory)
    return Chroma(
        collection_name=collection,
        embedding_function=emb,
        client_settings=client_settings,
    )

def init_uploaded_store(cfg) -> Chroma:
    p = _vec_dir_for(cfg, "uploaded")
    return _new_chroma("uploaded_docs", p, _embeddings())

def init_sharepoint_store(cfg) -> Chroma:
    p = _vec_dir_for(cfg, "sharepoint")
    return _new_chroma("sharepoint_docs", p, _embeddings())

# -------------------- maintenance --------------------
def _wipe_dir(path: Path):
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    finally:
        path.mkdir(parents=True, exist_ok=True)

def wipe_up_store(cfg):
    _wipe_dir(_vec_dir_for(cfg, "uploaded"))

def wipe_sp_store(cfg):
    _wipe_dir(_vec_dir_for(cfg, "sharepoint"))

# -------------------- ingestion --------------------
def _persist_silent(store: Chroma):
    try:
        store.persist()
    except Exception:
        # in-memory backend or older chromadb without persist -> ignore
        pass

def ingest_files(paths: Iterable[str], store: Chroma, chunk_size: int = 1000) -> List[str]:
    """
    Extracts text, chunks it, adds to the given Chroma store, and persists if supported.
    Returns list of successfully ingested file paths.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
    )
    added: List[str] = []
    for p in paths:
        try:
            text = extract_text(p)
            if not text or not text.strip():
                continue
            chunks = splitter.split_text(text)
            metadatas = [{"source": p} for _ in chunks]
            store.add_texts(chunks, metadatas=metadatas)
            added.append(p)
        except Exception:
            continue
    _persist_silent(store)
    return added

def ingest_sharepoint(cfg, include_exts: Optional[List[str]] = None, include_folder_ids: Optional[List[str]] = None) -> List[str]:
    """
    Downloads files from SharePoint via your sharepoint.py helpers, indexes them into Chroma,
    and returns the list of local file paths that were ingested.
    """
    if include_exts is None:
        include_exts = ["pdf", "docx", "pptx"]

    if _sp_fetch is None and _sp_ingest_wrap is None:
        raise RuntimeError(
            "SharePoint fetcher not available. Ensure sharepoint.py is importable "
            "and exposes fetch_files_from_sharepoint or ingest_sharepoint_files."
        )

    # Prefer direct fetch; fall back to wrapper if needed
    if _sp_fetch is not None:
        local_files = _sp_fetch(cfg, include_exts=include_exts, include_folder_ids=include_folder_ids) or []
    else:
        local_files = _sp_ingest_wrap(cfg, include_exts=include_exts, include_folder_ids=include_folder_ids) or []

    if not local_files:
        return []

    store = init_sharepoint_store(cfg)
    splitter = _text_splitter(cfg)

    for local_path in local_files:
        try:
            text = extract_text(local_path)
            if not text or not text.strip():
                continue
            chunks = splitter.split_text(text)
            # You can extend metadata here (e.g., detected category) if desired.
            metadatas = [{"source": local_path} for _ in chunks]
            store.add_texts(chunks, metadatas=metadatas)
        except Exception:
            continue

    _persist_silent(store)
    return local_files
