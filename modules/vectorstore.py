# modules/vectorstore.py
# -----------------------------------------------------------------------------
# Vector store utilities with robust backends:
# - Uploaded docs  : in-memory Chroma (no SQLite dependency)
# - SharePoint     : persistent Chroma (self-healing), FAISS fallback if needed
# - Normalized embeddings for consistent cosine similarity across backends
# - Page-aware PDF ingestion + logging + warnings on empty texts
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

# Logger
log = logging.getLogger(__name__)

# Embeddings & splitters
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Prefer Chroma; may be unavailable in some envs
_CHROMA_AVAILABLE = False
try:
    from langchain_community.vectorstores import Chroma
    from chromadb.config import Settings as _ChromaSettings
    _CHROMA_AVAILABLE = True
except Exception:
    _CHROMA_AVAILABLE = False

# FAISS fallback (no sqlite)
_FAISS_AVAILABLE = False
try:
    from langchain_community.vectorstores import FAISS
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False

# ---- Robust local imports (project structure can be root-level or modules/*)
try:
    from text_extraction import extract_text
except ModuleNotFoundError:
    from modules.text_extraction import extract_text  # type: ignore

# Optional: per-page PDF text (if available)
try:
    from modules.text_extraction import pdf_text_with_pages as _pdf_pages  # type: ignore
except Exception:
    _pdf_pages = None

# SharePoint fetchers: support either export names or wrapper
_sp_fetch = _sp_ingest_wrap = None
try:
    from sharepoint import fetch_files_from_sharepoint as _sp_fetch  # type: ignore
    from sharepoint import ingest_sharepoint_files as _sp_ingest_wrap  # type: ignore
except ModuleNotFoundError:
    try:
        from modules.sharepoint import fetch_files_from_sharepoint as _sp_fetch  # type: ignore
    except Exception:
        _sp_fetch = None
    try:
        from modules.sharepoint import ingest_sharepoint_files as _sp_ingest_wrap  # type: ignore
    except Exception:
        _sp_ingest_wrap = None

# ---- Defaults / constants
DEF_BASE = Path(".")
DEF_VEC_DIR = DEF_BASE / "vectorstore"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =============================================================================
# Helpers
# =============================================================================
def _vec_dir_for(cfg, kind: str) -> Path:
    """
    Returns a persistent directory for the given store kind.
    - 'uploaded'   -> ./vectorstore/uploaded   (only used for FAISS fallback)
    - 'sharepoint' -> ./vectorstore/sharepoint
    """
    base = Path(getattr(cfg, "VECTORSTORE_DIR", DEF_VEC_DIR))
    p = base / ("uploaded" if kind == "uploaded" else "sharepoint")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _embeddings():
    # Normalize embeddings so cosine behaves consistently across FAISS/Chroma
    return HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _text_splitter(cfg):
    return RecursiveCharacterTextSplitter(
        chunk_size=getattr(cfg, "CHUNK_SIZE", 1000),
        chunk_overlap=getattr(cfg, "CHUNK_OVERLAP", 120),
        separators=["\n\n", "\n", " ", ""],
    )


def _sqlite_is_new_enough() -> bool:
    """
    Check stdlib sqlite version; Chroma wants >= 3.35.0
    """
    try:
        import sqlite3
        parts = [int(x) for x in getattr(sqlite3, "sqlite_version", "0.0.0").split(".")[:3]]
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts) >= (3, 35, 0)
    except Exception:
        return False


# =============================================================================
# FAISS adapter (mirror the subset of VectorStore API we use)
# =============================================================================
class _FaissStoreAdapter:
    """
    Wrap FAISS to look like the pieces of the LangChain VectorStore we rely on:
      - add_texts(texts, metadatas)
      - persist() (saves index.faiss + index.pkl)
      - similarity_search(query, k) [optional]
      - as_retriever(**kwargs) [optional]
      - get_all_texts() for keyword fallback in retrieval
    """

    def __init__(self, dir_path: Path, embedding):
        self._dir = Path(dir_path)
        self._emb = embedding
        self._store = None
        self._texts: List[str] = []
        self._metas: List[dict] = []
        try:
            if (self._dir / "index.faiss").exists():
                self._store = FAISS.load_local(
                    str(self._dir),
                    self._emb,
                    allow_dangerous_deserialization=True,
                )
                log.info("[VectorStore] FAISS loaded from %s", str(self._dir))
        except Exception as ex:
            log.warning("[VectorStore] FAISS load failed: %s", ex)
            self._store = None

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        if metadatas is None:
            metadatas = [{} for _ in texts]
        # Keep sidecar copies for keyword fallback
        self._texts.extend(texts)
        self._metas.extend(metadatas)

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

    def get_all_texts(self) -> List[tuple[str, dict]]:
        return list(zip(self._texts, self._metas))


# =============================================================================
# Chroma initializers
# =============================================================================
def _init_uploaded_inmemory_chroma(emb):
    """
    Build an in-memory Chroma store for uploaded docs.
    - No SQLite (persist_directory=None, is_persistent=False)
    """
    settings = _ChromaSettings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=False,   # in-memory
        # NOTE: DO NOT set persist_directory at all for in-memory
    )
    return Chroma(
        collection_name="uploaded_docs_session",
        embedding_function=emb,
        client_settings=settings,
    )


def _build_chroma_or_reset(vec_dir: Path, collection_name: str, emb):
    """
    Persistent Chroma builder for SharePoint store.
    If the underlying SQLite DB is corrupted / partial (e.g., 'no such table ...'),
    it wipes the folder and recreates cleanly.
    """
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
        # Touch the store so tables exist
        vs.persist()
        return vs
    except Exception as e:
        msg = str(e).lower()
        if "no such table" in msg or "internalerror" in msg or "database error" in msg:
            # Reset broken schema
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
        raise


# =============================================================================
# Store factories
# =============================================================================
def _init_store(kind: str, cfg):
    """
    Factory for vector stores:
      - 'uploaded'   -> in-memory Chroma (no SQLite). If Chroma missing, FAISS as last resort.
      - 'sharepoint' -> persistent Chroma (self-heal) when sqlite>=3.35; else FAISS.
    """
    emb = _embeddings()

    if kind == "uploaded":
        if _CHROMA_AVAILABLE:
            try:
                vs = _init_uploaded_inmemory_chroma(emb)
                log.info("[VectorStore] uploaded backend = Chroma (in-memory)")
                return vs
            except Exception as ex:
                log.warning("[VectorStore] uploaded Chroma init failed: %s (falling back)", ex)
        if _FAISS_AVAILABLE:
            log.info("[VectorStore] uploaded backend = FAISS (local)")
            return _FaissStoreAdapter(_vec_dir_for(cfg, "uploaded"), emb)
        raise RuntimeError("No vector backend available for uploaded docs (need Chroma or FAISS).")

    # SharePoint (persistent)
    vec_dir = _vec_dir_for(cfg, "sharepoint")
    if _CHROMA_AVAILABLE and _sqlite_is_new_enough():
        try:
            vs = _build_chroma_or_reset(vec_dir, "sharepoint_docs", emb)
            log.info("[VectorStore] sharepoint backend = Chroma (persistent)")
            return vs
        except Exception as ex:
            log.warning("[VectorStore] sharepoint Chroma init failed: %s (falling back)", ex)
    if _FAISS_AVAILABLE:
        log.info("[VectorStore] sharepoint backend = FAISS (persistent)")
        return _FaissStoreAdapter(vec_dir, emb)
    raise RuntimeError("Vector store unavailable: Chroma requires sqlite>=3.35 and FAISS is not installed.")


def init_uploaded_store(cfg):
    return _init_store("uploaded", cfg)


def init_sharepoint_store(cfg):
    return _init_store("sharepoint", cfg)


# =============================================================================
# Maintenance
# =============================================================================
def wipe_up_store(cfg):
    """Remove the uploaded store directory (only relevant if FAISS was used)."""
    p = _vec_dir_for(cfg, "uploaded")
    if p.exists():
        for f in p.glob("**/*"):
            try:
                f.unlink()
            except Exception:
                pass
        try:
            p.rmdir()
        except Exception:
            pass


def wipe_sp_store(cfg):
    """Remove the SharePoint store directory (Chroma/FAISS persistent)."""
    p = _vec_dir_for(cfg, "sharepoint")
    if p.exists():
        for f in p.glob("**/*"):
            try:
                f.unlink()
            except Exception:
                pass
        try:
            p.rmdir()
        except Exception:
            pass


# =============================================================================
# Ingestion
# =============================================================================
def ingest_files(paths: Iterable[str], store, chunk_size: int = 1000) -> List[str]:
    """
    Ingest a list of local files into the provided store (uploaded docs flow).
    Works with both Chroma and FAISS adapter.
    - PDF: page-aware if pdf_text_with_pages is available
    - Logs warnings when no text could be extracted (e.g., scanned PDF without OCR)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
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
                text = extract_text(p) or ""
                if not text.strip():
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


def _derive_sp_category(path_like: str) -> str:
    rp = (path_like or "").lower()
    if "staff augmentation-it professional services" in rp:
        return "Staff Augmentation-IT Professional Services"
    return "Other"


def ingest_sharepoint(cfg, include_exts: Optional[List[str]] = None) -> List[str]:
    """
    Download from cfg.SP_SITE_URL and index into the SharePoint store.
    Uses persistent Chroma (self-healing) when possible; else FAISS.
    - PDF: page-aware if pdf_text_with_pages is available
    - Logs warnings when no text could be extracted
    """
    if include_exts is None:
        include_exts = ["pdf", "docx", "pptx"]

    if _sp_fetch is None and _sp_ingest_wrap is None:
        raise RuntimeError(
            "SharePoint fetcher not available. Ensure sharepoint.py exposes "
            "fetch_files_from_sharepoint or ingest_sharepoint_files."
        )

    # Pull files
    if _sp_fetch is not None:
        local_files = _sp_fetch(cfg, include_exts=include_exts) or []
    else:
        local_files = _sp_ingest_wrap(cfg, include_exts=include_exts) or []

    if not local_files:
        log.info("[SP Ingest] No files fetched for ingestion.")
        return []

    # Index
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
                text = extract_text(local_path) or ""
                if not text.strip():
                    log.warning("[SP Ingest] No text: %s", local_path)
                    continue
                chunks = splitter.split_text(text)
                metadatas = [{"source": local_path, "sp_category": cat} for _ in chunks]
                sp_store.add_texts(chunks, metadatas=metadatas)
            sp_store.persist()
        except Exception as ex:
            log.warning("[SP Ingest] Skipped %s: %s", local_path, ex)
            continue

    return local_files
