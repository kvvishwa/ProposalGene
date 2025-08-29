# modules/vectorstore.py
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pickle

# robust local imports (both flat and package)
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


# -------------------- paths --------------------
def _base_dir(cfg) -> Path:
    # Always writable on Azure:
    # uses env BASE_DIR if provided; else cfg.BASE_DIR; else /home/site/wwwroot/data; else ./data
    env_base = os.getenv("BASE_DIR")
    if env_base:
        p = Path(env_base)
    else:
        p = Path(getattr(cfg, "BASE_DIR", "/home/site/wwwroot/data"))
    p.mkdir(parents=True, exist_ok=True)
    return p

def _vec_root(cfg) -> Path:
    # Keep vectors under the data root
    root = _base_dir(cfg) / "vectorstore_faiss"
    root.mkdir(parents=True, exist_ok=True)
    return root

def _vec_dir_for(cfg, kind: str) -> Path:
    sub = "uploaded" if kind == "uploaded" else "sharepoint"
    p = _vec_root(cfg) / sub
    p.mkdir(parents=True, exist_ok=True)
    return p

def _faiss_paths(folder: Path):
    index_path = folder / "faiss.index"
    store_path = folder / "docstore.pkl"
    return index_path, store_path


# -------------------- embeddings & splitter --------------------
def _embeddings() -> OpenAIEmbeddings:
    # No Torch needed. Set OPENAI_API_KEY in Azure App Settings.
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)

def _splitter(cfg) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=getattr(cfg, "CHUNK_SIZE", 1000),
        chunk_overlap=getattr(cfg, "CHUNK_OVERLAP", 120),
        separators=["\n\n", "\n", " ", ""],
    )


# -------------------- store load/save --------------------
def _load_faiss(folder: Path) -> Optional[FAISS]:
    index_path, store_path = _faiss_paths(folder)
    emb = _embeddings()
    try:
        if index_path.exists() and store_path.exists():
            with open(store_path, "rb") as f:
                docstore = pickle.load(f)
            # FAISS.load_local expects folder save; use LC helper:
            return FAISS.load_local(
                folder_path=str(folder),
                embeddings=emb,
                allow_dangerous_deserialization=True,
            )
    except Exception:
        pass
    return None

def _save_faiss(store: FAISS, folder: Path) -> None:
    try:
        store.save_local(str(folder))
    except Exception:
        pass


# -------------------- singletons --------------------
_g_up: Optional[FAISS] = None
_g_sp: Optional[FAISS] = None


def init_uploaded_store(cfg) -> FAISS:
    global _g_up
    folder = _vec_dir_for(cfg, "uploaded")
    if _g_up is None:
        _g_up = _load_faiss(folder)
        if _g_up is None:
            _g_up = FAISS.from_texts([" "], _embeddings())  # seed empty index
            _save_faiss(_g_up, folder)
    return _g_up

def init_sharepoint_store(cfg) -> FAISS:
    global _g_sp
    folder = _vec_dir_for(cfg, "sharepoint")
    if _g_sp is None:
        _g_sp = _load_faiss(folder)
        if _g_sp is None:
            _g_sp = FAISS.from_texts([" "], _embeddings())  # seed empty index
            _save_faiss(_g_sp, folder)
    return _g_sp


# -------------------- maintenance --------------------
def _wipe_dir(path: Path):
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    finally:
        path.mkdir(parents=True, exist_ok=True)

def wipe_up_store(cfg):
    global _g_up
    _wipe_dir(_vec_dir_for(cfg, "uploaded"))
    _g_up = None

def wipe_sp_store(cfg):
    global _g_sp
    _wipe_dir(_vec_dir_for(cfg, "sharepoint"))
    _g_sp = None


# -------------------- ingestion --------------------
def ingest_files(paths: Iterable[str], store: FAISS, cfg, kind: str) -> List[str]:
    """
    Extract text, chunk, add to FAISS, and persist.
    'kind' must be 'uploaded' or 'sharepoint' for correct save path.
    """
    splitter = _splitter(cfg)
    docs: List[Document] = []
    added: List[str] = []

    for pth in paths:
        try:
            txt = extract_text(pth) or ""
            if not txt.strip():
                continue
            for chunk in splitter.split_text(txt):
                docs.append(Document(page_content=chunk, metadata={"source": pth}))
            added.append(pth)
        except Exception:
            continue

    if docs:
        store.add_documents(docs)
        # persist
        folder = _vec_dir_for(cfg, kind)
        _save_faiss(store, folder)

    return added


def ingest_sharepoint(cfg, include_exts: Optional[List[str]] = None, include_folder_ids: Optional[List[str]] = None) -> List[str]:
    if include_exts is None:
        include_exts = ["pdf", "docx", "pptx"]

    if _sp_fetch is None and _sp_ingest_wrap is None:
        raise RuntimeError("SharePoint fetcher not available (sharepoint.py not importable).")

    if _sp_fetch is not None:
        files = _sp_fetch(cfg, include_exts=include_exts, include_folder_ids=include_folder_ids) or []
    else:
        files = _sp_ingest_wrap(cfg, include_exts=include_exts, include_folder_ids=include_folder_ids) or []

    if not files:
        return []

    store = init_sharepoint_store(cfg)
    _ = ingest_files(files, store, cfg, kind="sharepoint")
    return files
