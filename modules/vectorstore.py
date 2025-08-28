# modules/vectorstore.py
from __future__ import annotations
from pathlib import Path
import json, os, shutil
from typing import Iterable, List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from modules.text_extraction import extract_text
from modules.sharepoint import ingest_sharepoint_files

# -------------------- singletons --------------------
g_sp_store: Optional[InMemoryVectorStore] = None
g_up_store: Optional[InMemoryVectorStore] = None

# -------------------- config helpers --------------------
def _base_dir(cfg) -> Path:
    p = Path(getattr(cfg, "BASE_DIR", "."))
    p.mkdir(parents=True, exist_ok=True)
    return p

def _col_dir(cfg, name: str) -> Path:
    d = _base_dir(cfg) / name
    d.mkdir(parents=True, exist_ok=True)
    return d

def _index_json_path(cfg, name: str) -> Path:
    return _col_dir(cfg, name) / "index.jsonl"

def _embeddings() -> OpenAIEmbeddings:
    # Uses OPENAI_API_KEY from env; allow override of model if you like
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)

# -------------------- persistence (texts + metadata only) --------------------
def _load_docs_from_disk(cfg, collection: str) -> List[Document]:
    p = _index_json_path(cfg, collection)
    docs: List[Document] = []
    if not p.exists():
        return docs
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                docs.append(Document(page_content=rec["text"], metadata=rec.get("metadata", {})))
            except Exception:
                continue
    return docs

def _append_docs_to_disk(cfg, collection: str, docs: List[Document]) -> None:
    if not docs:
        return
    p = _index_json_path(cfg, collection)
    with p.open("a", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d.page_content, "metadata": d.metadata}, ensure_ascii=False) + "\n")

# -------------------- store lifecycle --------------------
def _build_store_from_docs(docs: List[Document]) -> InMemoryVectorStore:
    emb = _embeddings()
    if not docs:
        # create an empty store
        return InMemoryVectorStore(embedding=emb)
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    return InMemoryVectorStore.from_texts(texts=texts, metadatas=metadatas, embedding=emb)

def init_sharepoint_store(cfg):
    global g_sp_store
    if g_sp_store is None:
        docs = _load_docs_from_disk(cfg, cfg.SP_COLLECTION)
        g_sp_store = _build_store_from_docs(docs)
    return g_sp_store

def init_uploaded_store(cfg):
    global g_up_store
    if g_up_store is None:
        docs = _load_docs_from_disk(cfg, cfg.UP_COLLECTION)
        g_up_store = _build_store_from_docs(docs)
    return g_up_store

# -------------------- ingestion --------------------
def ingest_files(files: Iterable[str], store: InMemoryVectorStore, chunk_size: int) -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    new_docs: List[Document] = []

    for path in files:
        try:
            text = extract_text(path) or ""
        except Exception:
            text = ""
        if not text.strip():
            continue
        for chunk in splitter.split_text(text):
            new_docs.append(Document(page_content=chunk, metadata={"source": path}))

    if not new_docs:
        return

    # Persist texts+metadata first
    # Decide which collection this store represents by checking both singletons
    from_langchain = (store is g_up_store, store is g_sp_store)
    if from_langchain[0]:
        _append_docs_to_disk(_cfg_like(), _cfg_like().UP_COLLECTION, new_docs)
    elif from_langchain[1]:
        _append_docs_to_disk(_cfg_like(), _cfg_like().SP_COLLECTION, new_docs)
    else:
        # Default to uploaded collection if not matched
        _append_docs_to_disk(_cfg_like(), _cfg_like().UP_COLLECTION, new_docs)

    # Add to in-memory index
    store.add_documents(new_docs)

def ingest_sharepoint(cfg, include_exts=None, include_folder_ids=None):
    files = ingest_sharepoint_files(cfg, include_exts=include_exts, include_folder_ids=include_folder_ids)
    if not files:
        return []
    sp_store = init_sharepoint_store(cfg)
    ingest_files(files, sp_store, cfg.CHUNK_SIZE)
    return files

# -------------------- wipe --------------------
def _wipe_dir(path: Path):
    try:
        if path.exists():
            shutil.rmtree(path)
    except Exception:
        pass

def wipe_sp_store(cfg):
    _wipe_dir(_col_dir(cfg, cfg.SP_COLLECTION))
    global g_sp_store
    g_sp_store = None

def wipe_up_store(cfg):
    _wipe_dir(_col_dir(cfg, cfg.UP_COLLECTION))
    global g_up_store
    g_up_store = None

# -------------------- tiny cfg proxy (for ingest_files persistence routing) --------------------
class _CfgProxy:
    def __init__(self):
        self.BASE_DIR = os.getenv("BASE_DIR", ".")
        self.UP_COLLECTION = os.getenv("UP_COLLECTION", "uploaded_docs")
        self.SP_COLLECTION = os.getenv("SP_COLLECTION", "sharepoint_docs")

def _cfg_like() -> _CfgProxy:
    return _CfgProxy()
