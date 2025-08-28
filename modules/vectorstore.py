# modules/vectorstore.py (lightweight, no sqlite, no faiss, no torch)
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

g_sp_store = None
g_up_store = None

def _store_dir(base_dir: str, name: str) -> Path:
    p = Path(base_dir) / name
    p.mkdir(parents=True, exist_ok=True)
    return p

def _load_or_init_docarray(base_dir: str, collection: str) -> DocArrayHnswSearch:
    emb = OpenAIEmbeddings()  # uses OPENAI_API_KEY from env
    dst = _store_dir(base_dir, collection)
    try:
        return DocArrayHnswSearch.load(dst, embeddings=emb)
    except Exception:
        # create an empty index
        return DocArrayHnswSearch.from_texts([" "], emb)

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
    if docs:
        store.add_documents(docs)

def ingest_sharepoint(cfg, include_exts=None, include_folder_ids=None):
    files = ingest_sharepoint_files(cfg, include_exts=include_exts, include_folder_ids=include_folder_ids)
    if not files:
        return []
    sp_store = init_sharepoint_store(cfg)
    ingest_files(files, sp_store, cfg.CHUNK_SIZE)
    # persist index
    sp_store.save(_store_dir(cfg.BASE_DIR, cfg.SP_COLLECTION))
    return files

def wipe_sp_store(cfg):
    base = Path(cfg.BASE_DIR) / cfg.SP_COLLECTION
    try:
        if base.exists(): shutil.rmtree(base)
    except Exception:
        pass
    global g_sp_store; g_sp_store = None

def wipe_up_store(cfg):
    base = Path(cfg.BASE_DIR) / cfg.UP_COLLECTION
    try:
        if base.exists(): shutil.rmtree(base)
    except Exception:
        pass
    global g_up_store; g_up_store = None
