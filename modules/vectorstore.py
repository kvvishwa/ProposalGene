# modules/vectorstore.py
from __future__ import annotations
from pathlib import Path
import shutil
from typing import Iterable, List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayHnswSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from modules.text_extraction import extract_text
from modules.sharepoint import ingest_sharepoint_files


# --- Singletons ---
g_sp_store: DocArrayHnswSearch | None = None
g_up_store: DocArrayHnswSearch | None = None


# --- Internal helpers ---
def _store_dir(base_dir: str | Path, name: str) -> Path:
    """Ensure and return the on-disk folder for the index."""
    p = Path(base_dir) / name
    p.mkdir(parents=True, exist_ok=True)
    return p

def _make_embeddings(model: str | None = None) -> OpenAIEmbeddings:
    # Uses OPENAI_API_KEY from env; model param is ignored for OpenAIEmbeddings
    return OpenAIEmbeddings()

def _load_or_init_docarray(base_dir: str | Path, collection: str) -> DocArrayHnswSearch:
    """Load an existing DocArrayHnswSearch index, or create a new one with a placeholder text."""
    emb = _make_embeddings()
    wd = _store_dir(base_dir, collection)

    # Try load existing
    try:
        store = DocArrayHnswSearch.load(str(wd), embeddings=emb)
        # attach for later saves
        setattr(store, "_work_dir", str(wd))
        return store
    except Exception:
        pass

    # Create minimal new index with an innocuous placeholder
    store = DocArrayHnswSearch.from_texts([" "], emb, work_dir=str(wd))
    setattr(store, "_work_dir", str(wd))
    return store

def _save(store: DocArrayHnswSearch):
    """Persist the index back to disk."""
    wd = getattr(store, "_work_dir", None)
    if wd:
        store.save(wd)


# --- Public init entrypoints ---
def init_sharepoint_store(cfg):
    global g_sp_store
    if g_sp_store is None:
        g_sp_store = _load_or_init_docarray(cfg.BASE_DIR, cfg.SP_COLLECTION)
    return g_sp_store

def init_uploaded_store(cfg):
    global g_up_store
    if g_up_store is None:
        g_up_store = _load_or_init_docarray(cfg.BASE_DIR, cfg.UP_COLLECTION)
    return g_up_store


# --- Ingestion ---
def ingest_files(files: Iterable[str], store: DocArrayHnswSearch, chunk_size: int) -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    docs: List[Document] = []

    for path in files:
        try:
            text = extract_text(path) or ""
        except Exception:
            text = ""
        if not text.strip():
            continue

        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": path}))

    if not docs:
        return

    store.add_documents(docs)
    _save(store)  # persist after adding docs

def ingest_sharepoint(cfg, include_exts=None, include_folder_ids=None):
    files = ingest_sharepoint_files(cfg, include_exts=include_exts, include_folder_ids=include_folder_ids)
    if not files:
        return []
    sp_store = init_sharepoint_store(cfg)
    ingest_files(files, sp_store, cfg.CHUNK_SIZE)  # _save() happens inside
    return files


# --- Maintenance ---
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
