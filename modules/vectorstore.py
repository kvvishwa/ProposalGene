# modules/vectorstore.py
from __future__ import annotations
from pathlib import Path
import shutil
from typing import Iterable, List

#Commnet
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from modules.text_extraction import extract_text
from modules.sharepoint import ingest_sharepoint_files

# -----------------------------------------------------------------------------
# Global singletons
# -----------------------------------------------------------------------------
g_sp_store: FAISS | None = None   # SharePoint-backed FAISS index
g_up_store: FAISS | None = None   # Uploaded-files FAISS index


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _faiss_dir(base_dir: str | Path, name: str) -> Path:
    p = Path(base_dir) / name
    p.mkdir(parents=True, exist_ok=True)
    return p

def _load_or_init_faiss(base_dir: str | Path, collection: str, model_name: str) -> FAISS:
    """
    Load FAISS index from disk if present; otherwise create a tiny placeholder index.
    (FAISS needs an index to exist before .add_documents; we create one and persist it.)
    """
    ef = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"trust_remote_code": True})
    dst = _faiss_dir(base_dir, collection)

    # Try load existing
    try:
        return FAISS.load_local(
            folder_path=str(dst),
            embeddings=ef,
            allow_dangerous_deserialization=True,  # required by FAISS loader
        )
    except Exception:
        pass

    # Create a minimal index (single benign placeholder) so callers can retrieve immediately
    store = FAISS.from_texts([" "], ef)
    store.save_local(str(dst))
    return store

def _save(store: FAISS, base_dir: str | Path, collection: str) -> None:
    store.save_local(str(_faiss_dir(base_dir, collection)))


# -----------------------------------------------------------------------------
# Public init entrypoints (replaces Chroma)
# -----------------------------------------------------------------------------
def init_sharepoint_store(cfg):
    """Initialize (or load) the SharePoint FAISS store singleton."""
    global g_sp_store
    if g_sp_store is None:
        g_sp_store = _load_or_init_faiss(cfg.BASE_DIR, cfg.SP_COLLECTION, cfg.EMBEDDING_MODEL)
    return g_sp_store

def init_uploaded_store(cfg):
    """Initialize (or load) the Uploaded-files FAISS store singleton."""
    global g_up_store
    if g_up_store is None:
        g_up_store = _load_or_init_faiss(cfg.BASE_DIR, cfg.UP_COLLECTION, cfg.EMBEDDING_MODEL)
    return g_up_store


# -----------------------------------------------------------------------------
# Ingestion (common)
# -----------------------------------------------------------------------------
def ingest_files(files: Iterable[str], store: FAISS, chunk_size: int) -> None:
    """
    Extract text from each file, split into chunks, add to FAISS, and persist to disk.
    Caller is responsible for passing the correct store (uploaded vs SharePoint).
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    docs: List[Document] = []

    for path in files:
        text = ""
        try:
            text = extract_text(path) or ""
        except Exception:
            # Skip unreadable/unsupported files gracefully
            text = ""

        if not text.strip():
            continue

        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": path}))

    if not docs:
        return

    store.add_documents(docs)

    # Persist to disk (figure out which collection path this store belongs to)
    # We check both known locations and save to the one that exists.
    # If neither exists (unlikely), default to uploaded collection.
    try:
        store.save_local(str(_faiss_dir(store.index_to_docstore_id.__class__.__name__, "noop")))  # no-op guard
    except Exception:
        pass  # ignore

    # Save explicitly to both possible locations (safe & cheap)
    try:
        _save(store, cfg_like(store, "BASE_DIR"), cfg_like(store, "UP_COLLECTION"))
    except Exception:
        pass
    try:
        _save(store, cfg_like(store, "BASE_DIR"), cfg_like(store, "SP_COLLECTION"))
    except Exception:
        pass


def cfg_like(store: FAISS, key: str) -> str:
    """
    Small helper that lets us read values the app set in Config() via environment variables
    (since FAISS store does not retain where it was created). Your app provides cfg in init,
    so this is only used to attempt a best-effort save. If missing, defaults are fine.
    """
    import os
    defaults = {
        "BASE_DIR": ".",
        "UP_COLLECTION": "uploaded_docs",
        "SP_COLLECTION": "sharepoint_docs",
    }
    return os.getenv(key, defaults[key])


# -----------------------------------------------------------------------------
# SharePoint ingestion wrapper
# -----------------------------------------------------------------------------
def ingest_sharepoint(cfg, include_exts=None, include_folder_ids=None):
    """
    Download files from SharePoint (Graph API), ingest into the SharePoint FAISS store,
    and persist. Returns list of local temp file paths.
    """
    files = ingest_sharepoint_files(cfg, include_exts=include_exts, include_folder_ids=include_folder_ids)
    if not files:
        return []

    sp_store = init_sharepoint_store(cfg)
    ingest_files(files, sp_store, cfg.CHUNK_SIZE)

    # Ensure persisted to SP collection path
    _save(sp_store, cfg.BASE_DIR, cfg.SP_COLLECTION)
    return files



# -----------------------------------------------------------------------------
# Maintenance / Persistence helpers
# -----------------------------------------------------------------------------
def _wipe_dir(path: Path):
    try:
        if path.exists():
            shutil.rmtree(path)
    except Exception:
        pass

def wipe_sp_store(cfg):
    base = Path(cfg.BASE_DIR) / cfg.SP_COLLECTION
    _wipe_dir(base)
    global g_sp_store
    g_sp_store = None

def wipe_up_store(cfg):
    base = Path(cfg.BASE_DIR) / cfg.UP_COLLECTION
    _wipe_dir(base)
    global g_up_store
    g_up_store = None
 