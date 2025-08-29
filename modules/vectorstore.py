# modules/vectorstore.py
from __future__ import annotations

import os
import shutil
import pickle
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ---------- robust local imports (flat or package) ----------
try:
    from text_extraction import extract_text
except ModuleNotFoundError:
    from modules.text_extraction import extract_text  # fallback

# We support either helper name exposed by your sharepoint.py
try:
    from sharepoint import fetch_files_from_sharepoint as _sp_fetch
except ModuleNotFoundError:
    try:
        from modules.sharepoint import fetch_files_from_sharepoint as _sp_fetch
    except Exception:
        _sp_fetch = None

try:
    from sharepoint import ingest_sharepoint_files as _sp_ingest_wrap
except ModuleNotFoundError:
    try:
        from modules.sharepoint import ingest_sharepoint_files as _sp_ingest_wrap
    except Exception:
        _sp_ingest_wrap = None


# ============================ Paths & Embeddings ============================

def _base_dir(cfg) -> Path:
    """
    Writable base dir on Azure:
      1) env BASE_DIR
      2) cfg.BASE_DIR
      3) /home/site/wwwroot/data (Azure)
    """
    env_base = os.getenv("BASE_DIR")
    if env_base:
        p = Path(env_base)
    else:
        p = Path(getattr(cfg, "BASE_DIR", "/home/site/wwwroot/data"))
    p.mkdir(parents=True, exist_ok=True)
    return p

def _vec_root(cfg) -> Path:
    root = _base_dir(cfg) / "vectorstore_faiss"
    root.mkdir(parents=True, exist_ok=True)
    return root

def _vec_dir_for(cfg, kind: str) -> Path:
    """kind = 'uploaded' | 'sharepoint'"""
    sub = "uploaded" if kind == "uploaded" else "sharepoint"
    p = _vec_root(cfg) / sub
    p.mkdir(parents=True, exist_ok=True)
    return p

def _faiss_paths(folder: Path):
    index_path = folder / "faiss.index"
    store_path = folder / "docstore.pkl"
    return index_path, store_path

def _embeddings() -> OpenAIEmbeddings:
    # No Torch. Set OPENAI_API_KEY in App Settings.
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)

def _splitter(chunk_size: int, chunk_overlap: int = 120) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )


# ============================ Save / Load ============================

def _load_faiss(folder: Path) -> Optional[FAISS]:
    emb = _embeddings()
    try:
        # LangChain FAISS supports folder-level save/load
        return FAISS.load_local(
            folder_path=str(folder),
            embeddings=emb,
            allow_dangerous_deserialization=True,
        )
    except Exception:
        # Fall back to manual check for legacy files then let create path
        index_path, store_path = _faiss_paths(folder)
        if index_path.exists() and store_path.exists():
            try:
                return FAISS.load_local(
                    folder_path=str(folder),
                    embeddings=emb,
                    allow_dangerous_deserialization=True,
                )
            except Exception:
                return None
    return None

def _save_faiss(store: FAISS, folder: Path) -> None:
    try:
        store.save_local(str(folder))
    except Exception:
        pass


# ============================ Singletons ============================

_g_up: Optional[FAISS] = None
_g_sp: Optional[FAISS] = None

def init_uploaded_store(cfg) -> FAISS:
    global _g_up
    folder = _vec_dir_for(cfg, "uploaded")
    if _g_up is None:
        _g_up = _load_faiss(folder)
        if _g_up is None:
            # Seed an empty index so .add_documents works immediately
            _g_up = FAISS.from_texts([" "], _embeddings())
            _save_faiss(_g_up, folder)
    return _g_up

def init_sharepoint_store(cfg) -> FAISS:
    global _g_sp
    folder = _vec_dir_for(cfg, "sharepoint")
    if _g_sp is None:
        _g_sp = _load_faiss(folder)
        if _g_sp is None:
            _g_sp = FAISS.from_texts([" "], _embeddings())
            _save_faiss(_g_sp, folder)
    return _g_sp


# ============================ Maintenance ============================

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


# ============================ Ingestion ============================

def ingest_files(paths: Iterable[str], store: FAISS, chunk_size: int, kind: str = "uploaded") -> List[str]:
    """
    NOTE: Signature matches your main.py (3rd arg = chunk_size).
    kind: "uploaded" (default) or "sharepoint" controls save location.
    """
    sp = _splitter(chunk_size)
    docs: List[Document] = []
    added: List[str] = []

    for pth in paths:
        try:
            txt = extract_text(pth) or ""
            if not txt.strip():
                continue
            for chunk in sp.split_text(txt):
                docs.append(Document(page_content=chunk, metadata={"source": pth}))
            added.append(pth)
        except Exception:
            continue

    if docs:
        store.add_documents(docs)
        folder = _vec_dir_for(_CfgProxy(), kind)  # persist using env defaults if cfg not handy
        _save_faiss(store, folder)

    return added


def ingest_sharepoint(cfg, include_exts: Optional[List[str]] = None, include_folder_ids: Optional[List[str]] = None) -> List[str]:
    """
    Downloads files from SharePoint via sharepoint.py and indexes them into the SharePoint FAISS store.
    """
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
    _ = ingest_files(files, store, getattr(cfg, "CHUNK_SIZE", 1000), kind="sharepoint")
    return files


# ============================ Tiny cfg proxy ============================

class _CfgProxy:
    """
    Used only by ingest_files() when it needs a save path but the caller passed no cfg.
    Reads env vars to find the same folders your app uses.
    """
    def __init__(self):
        self.BASE_DIR = os.getenv("BASE_DIR", "/home/site/wwwroot/data")
