# modules/vectorstore.py  (self-healing + FAISS fallback)

import os
import shutil
from pathlib import Path
from typing import List, Optional, Iterable

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Prefer Chroma, but allow absence
_CHROMA_AVAILABLE = False
try:
    from langchain_community.vectorstores import Chroma
    from chromadb.config import Settings as _ChromaSettings
    _CHROMA_AVAILABLE = True
except Exception:
    _CHROMA_AVAILABLE = False

# FAISS fallback
_FAISS_AVAILABLE = False
try:
    from langchain_community.vectorstores import FAISS
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False

# --- robust local imports ---
try:
    from text_extraction import extract_text
except ModuleNotFoundError:
    from modules.text_extraction import extract_text  # fallback

# Try both layouts: root-level and modules/
_sp_fetch = _sp_ingest_wrap = None
try:
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

def _sqlite_is_new_enough() -> bool:
    try:
        import sqlite3
        parts = [int(x) for x in getattr(sqlite3, "sqlite_version", "0.0.0").split(".")[:3]]
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts) >= (3, 35, 0)
    except Exception:
        return False

# ---------- FAISS adapter (mirrors used API) ----------
class _FaissStoreAdapter:
    def __init__(self, dir_path: Path, embedding):
        self._dir = Path(dir_path)
        self._emb = embedding
        self._store = None
        try:
            if (self._dir / "index.faiss").exists():
                self._store = FAISS.load_local(str(self._dir), self._emb, allow_dangerous_deserialization=True)
        except Exception:
            self._store = None

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        if self._store is None:
            self._store = FAISS.from_texts(texts, embedding=self._emb, metadatas=metadatas)
        else:
            self._store.add_texts(texts, metadatas=metadatas)

    def persist(self):
        if self._store is not None:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._store.save_local(str(self._dir))

    # Optional helpers
    def similarity_search(self, query: str, k: int = 4):
        return [] if self._store is None else self._store.similarity_search(query, k=k)
    def as_retriever(self, **kwargs):
        return None if self._store is None else self._store.as_retriever(**kwargs)

# ---------- Chroma builder with self-heal ----------
def _build_chroma_or_reset(vec_dir: Path, collection_name: str, emb):
    """Try Chroma; if schema error ('no such table', etc.), wipe folder and recreate."""
    settings = _ChromaSettings(
        persist_directory=str(vec_dir),
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=True,
    )
    try:
        vs = Chroma(
            collection_name=collection_name,
            persist_directory=str(vec_dir),
            embedding_function=emb,
            client_settings=settings,
        )
        # Touch to ensure tables exist
        vs.persist()
        return vs
    except Exception as e:
        msg = str(e).lower()
        if "no such table" in msg or "internalerror" in msg or "database error" in msg:
            # Folder is corrupted / mismatched schema. Reset it.
            try:
                shutil.rmtree(vec_dir, ignore_errors=True)
            except Exception:
                pass
            vec_dir.mkdir(parents=True, exist_ok=True)
            vs = Chroma(
                collection_name=collection_name,
                persist_directory=str(vec_dir),
                embedding_function=emb,
                client_settings=settings,
            )
            vs.persist()
            return vs
        # Different failure -> bubble up
        raise

def _init_store(kind: str, cfg):
    """Chroma if possible; otherwise FAISS."""
    vec_dir = _vec_dir_for(cfg, kind)
    emb = _embeddings()
    if _CHROMA_AVAILABLE and _sqlite_is_new_enough():
        try:
            return _build_chroma_or_reset(vec_dir, "uploaded_docs" if kind == "uploaded" else "sharepoint_docs", emb)
        except Exception:
            # Last-resort: fallback to FAISS
            pass
    if not _FAISS_AVAILABLE:
        raise RuntimeError("Vector store unavailable: Chroma requires sqlite>=3.35 and FAISS is not installed.")
    return _FaissStoreAdapter(vec_dir, emb)

def init_uploaded_store(cfg):
    return _init_store("uploaded", cfg)

def init_sharepoint_store(cfg):
    return _init_store("sharepoint", cfg)

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

def ingest_files(paths: Iterable[str], store, chunk_size: int = 1000) -> List[str]:
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

def _derive_sp_category(path_like: str) -> str:
    rp = (path_like or "").lower()
    if "staff augmentation-it professional services" in rp:
        return "Staff Augmentation-IT Professional Services"
    return "Other"

def ingest_sharepoint(cfg, include_exts: Optional[List[str]] = None) -> List[str]:
    """
    Download files from cfg.SP_SITE_URL and index them.
    Uses Chroma (self-healing) if possible; else FAISS; same API.
    """
    if include_exts is None:
        include_exts = ["pdf", "docx", "pptx"]

    if _sp_fetch is None and _sp_ingest_wrap is None:
        raise RuntimeError(
            "SharePoint fetcher not available. Ensure sharepoint.py exposes "
            "fetch_files_from_sharepoint or ingest_sharepoint_files."
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
            sp_store.persist()
        except Exception:
            continue

    return local_files
