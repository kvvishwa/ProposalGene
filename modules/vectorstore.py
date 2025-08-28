# modules/vectorstore.py
from __future__ import annotations
from pathlib import Path
import os, shutil
from typing import Iterable, List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayHnswSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from modules.text_extraction import extract_text
from modules.sharepoint import ingest_sharepoint_files

# -------------------- singletons --------------------
g_sp_store: Optional[DocArrayHnswSearch] = None
g_up_store: Optional[DocArrayHnswSearch] = None

def _emb() -> OpenAIEmbeddings:
    # Uses OPENAI_API_KEY from env; allow model override via OPENAI_EMBEDDING_MODEL
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)

def _dir(base_dir: str | Path, name: str) -> Path:
    p = Path(base_dir) / name
    p.mkdir(parents=True, exist_ok=True)
    return p

def _load_or_seed(folder: Path) -> DocArrayHnswSearch:
    """
    Try to load an existing DocArrayHnswSearch index.
    If not present, seed a tiny index using a placeholder text and persist it.
    """
    emb = _emb()
    try:
        store = DocArrayHnswSearch.load_local(str(folder), embeddings=emb)
        setattr(store, "_save_path", str(folder))
        return store
    except Exception:
        pass

    # Seed a minimal index so we can add docs later; persist immediately
    # (Requires embedding call to succeed; if it doesn't, we let the exception surface)
    store = DocArrayHnswSearch.from_texts([" "], emb)
    setattr(store, "_save_path", str(folder))
    store.save_local(str(folder))
    return store

def _save(store: DocArrayHnswSearch):
    sp = getattr(store, "_save_path", None)
    if sp:
        store.save_local(sp)

# -------------------- public init --------------------
def init_sharepoint_store(cfg):
    global g_sp_store
    if g_sp_store is None:
        folder = _dir(cfg.BASE_DIR, cfg.SP_COLLECTION)
        g_sp_store = _load_or_seed(folder)
    return g_sp_store

def init_uploaded_store(cfg):
    global g_up_store
    if g_up_store is None:
        folder = _dir(cfg.BASE_DIR, cfg.UP_COLLECTION)
        g_up_store = _load_or_seed(folder)
    return g_up_store

# -------------------- ingestion --------------------
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
    _save(store)

def ingest_sharepoint(cfg, include_exts=None, include_folder_ids=None):
    files = ingest_sharepoint_files(cfg, include_exts=include_exts, include_folder_ids=include_folder_ids)
    if not files:
        return []
    sp_store = init_sharepoint_store(cfg)
    ingest_files(files, sp_store, cfg.CHUNK_SIZE)
    return files

# -------------------- maintenance --------------------
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
