# modules/url_store.py
from __future__ import annotations
import json, uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

DEFAULT_FILENAME = "sp_sources.json"

@dataclass
class SPSite:
    id: str
    name: str
    site_url: str
    folder_path: str = ""        # e.g. "Bid Management Documents/107-Technical Proposal"
    notes: str = ""
    last_ingested: Optional[str] = None

def _store_path(base_dir: str) -> Path:
    p = Path(base_dir) / DEFAULT_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def load_sources(base_dir: str) -> List[SPSite]:
    p = _store_path(base_dir)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [SPSite(**item) for item in data]
    except Exception:
        return []

def save_sources(base_dir: str, items: List[SPSite]) -> None:
    p = _store_path(base_dir)
    p.write_text(json.dumps([asdict(i) for i in items], indent=2), encoding="utf-8")

def add_source(base_dir: str, name: str, site_url: str, folder_path: str = "", notes: str = "") -> SPSite:
    items = load_sources(base_dir)
    item = SPSite(id=str(uuid.uuid4()), name=name.strip(), site_url=site_url.strip(), folder_path=folder_path.strip(), notes=notes.strip())
    items.append(item)
    save_sources(base_dir, items)
    return item

def update_source(base_dir: str, updated: SPSite) -> None:
    items = load_sources(base_dir)
    by_id = {i.id: i for i in items}
    if updated.id in by_id:
        by_id[updated.id] = updated
        save_sources(base_dir, list(by_id.values()))

def delete_source(base_dir: str, source_id: str) -> None:
    items = load_sources(base_dir)
    items = [i for i in items if i.id != source_id]
    save_sources(base_dir, items)
