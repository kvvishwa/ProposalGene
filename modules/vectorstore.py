# modules/vectorstore.py
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings as ChromaSettings  # ensures duckdb+parquet

# --- robust local imports (both "modules.*" and flat) ---
try:
    from text_extraction import extract_text
except ModuleNotFoundError:
    from modules.text_extraction import extract_text

try:
    from sharepoint import fetch_files_from_sharepoint as _sp_fetch
    from sharepoint import ingest_sharepoint_files as _sp_ingest_wrap
except ModuleNotFoundError:
    _sp_fetch = None
    _sp_ingest_wrap = None
    try:
        from modules.sharepoint import fetch_files_from_sharepoint as _sp_fetch
    except Exception:
        _sp_fetch = None
    try:
        from modules.sharepoint import ingest_sharepoint_files as _sp_ingest_wrap
    except Exception:
        _sp_ingest_wrap = None


# -------------------- config & paths --------------------
def _base_dir(cfg) -> Path:
    """
    Guaranteed-writable base dir on Azure.
    Falls back to env BASE_DIR, then /home/site/wwwroot/data, then ./data.
    """
    env_base = os.getenv("BASE_DIR")
    if env_base:
        p = Path(env_base)
    else:
        p = Path(getattr(cfg, "BASE_DIR", "/home/site/wwwroot/data"))
    p.mkdir(parents=True, exist_ok=True)
    return p

def _vec_root(cfg) -> Path:
    """
    Vector root: VECTORSTORE_DIR or BASE_DIR/'vectorstore'
    (keeps vectors co-located with other app data).
    """
    v = os.getenv("VECTORSTORE_DIR", "")
    root = Path(v) if v else _base_dir(cfg) / "vectorstore"
    root.mkdir(parents=True, exist_ok=True)
    return root

def _vec_dir_for(cfg, kind: str) -> Path:
    sub = "uploaded" if kind == "uploaded" else "sharepoint"
    p = _vec_root(cfg) / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


# -------------------- embeddings & splitter --------------------
def _embeddings() -> OpenAIEmbeddings:
    """
    Uses OpenAI embeddings (no Torch). Set OPENAI_API_KEY in Azure App Settings.
    Optionally override model via OPENAI_EMBEDDING_MODEL (default: text-embedding-3-small).
    """
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)

def _splitter(cfg) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=getattr(cfg, "CHUNK_SIZE", 1000),
        chunk_overlap=getattr(cfg, "CHUNK_OVERLAP", 120),
        separators=["\n\n", "\n", " ", ""],
    )


# -------------------- Chroma factories (DuckDB+Parquet) --------------------
def _new_chroma(collection: str, persist_dir: Path) -> Chroma:
    """
    Force DuckDB+Parquet backend (no sqlite).
    Falls back to in-memory if Settings can’t be applied (very rare).
    """
    try:
        settings = ChromaSettings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(persist_dir),
            is_persistent=True,
        )
        return Chroma(
            collection_name=collection,
            embedding_function=_embeddings(),
            persist_directory=str(persist_dir),
            client_settings=settings,
        )
    except Exception:
        # in-memory fallback; still works, just not persistent
        return Chroma(
            collection_name=collection,
            embedding_function=_embeddings(),
        )

def init_uploaded_store(cfg) -> Chroma:
    return _new_chroma("uploaded_docs", _vec_dir_for(cfg, "uploaded"))

def init_sharepoint_store(cfg) -> Chroma:
    return _new_chroma("sharepoint_docs", _vec_dir_for(cfg, "sharepoint"))


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
def ingest_files(paths: Iterable[str], store: Chroma, chunk_size: int = 1000) -> List[str]:
    added: List[str] = []
    split = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
    )
    for pth in paths:
        try:
            txt = extract_text(pth) or ""
            if not txt.strip():
                continue
            chunks = split.split_text(txt)
            store.add_texts(chunks, metadatas=[{"source": pth}] * len(chunks))
            added.append(pth)
        except Exception:
            continue
    try:
        store.persist()
    except Exception:
        pass
    return added

def ingest_sharepoint(cfg, include_exts: Optional[List[str]] = None, include_folder_ids: Optional[List[str]] = None) -> List[str]:
    if include_exts is None:
        include_exts = ["pdf", "docx", "pptx"]

    if _sp_fetch is None and _sp_ingest_wrap is None:
        raise RuntimeError("SharePoint fetcher not available (sharepoint.py not imported).")

    # Prefer direct fetch; fallback to wrapper
    files = []
    if _sp_fetch is not None:
        files = _sp_fetch(cfg, include_exts=include_exts, include_folder_ids=include_folder_ids) or []
    elif _sp_ingest_wrap is not None:
        files = _sp_ingest_wrap(cfg, include_exts=include_exts, include_folder_ids=include_folder_ids) or []

    if not files:
        return []

    store = init_sharepoint_store(cfg)
    split = _splitter(cfg)

    for local_path in files:
        try:
            txt = extract_text(local_path) or ""
            if not txt.strip():
                continue
            chunks = split.split_text(txt)
            store.add_texts(chunks, metadatas=[{"source": local_path}] * len(chunks))
        except Exception:
            continue

    try:
        store.persist()
    except Exception:
        pass
    return files
