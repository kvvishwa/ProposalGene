# modules/sp_sources.py
from __future__ import annotations
import json, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

@dataclass
class SPSite:
    id: str                # stable key (e.g., slug: contoso-sales)
    name: str              # friendly name
    url: str               # canonical https://<tenant>.sharepoint.com/sites/<site>
    default_folders: List[str] = None  # optional include list (e.g., ["Bid Management Documents/107-Technical Proposal"])
    file_types: List[str] = None       # e.g., ["pdf","docx","pptx"]
    includes: List[str] = None         # path substrings to include
    excludes: List[str] = None         # path substrings to exclude
    min_bytes: int = 0
    max_bytes: int = 0                 # 0 = no limit
    modified_since: str = ""           # ISO8601; empty = no filter

class SPSources:
    def __init__(self, base_dir: str):
        self.base = Path(base_dir)
        self.sites_path = self.base / "sp_sources.json"
        self.state_path = self.base / "sp_index_state.json"
        self.sites_path.parent.mkdir(parents=True, exist_ok=True)
        self._sites = self._load_sites()
        self._state = self._load_state()

    # ---------- persistence ----------
    def _load_sites(self) -> Dict[str, Any]:
        if self.sites_path.exists():
            return json.loads(self.sites_path.read_text(encoding="utf-8"))
        return {"active_id": "", "sites": {}}

    def _save_sites(self):
        self.sites_path.write_text(json.dumps(self._sites, indent=2), encoding="utf-8")

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {}

    def _save_state(self):
        self.state_path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    # ---------- sites CRUD ----------
    def list_sites(self) -> List[SPSite]:
        return [SPSite(**v) for v in self._sites.get("sites", {}).values()]

    def get(self, site_id: str) -> Optional[SPSite]:
        d = self._sites.get("sites", {}).get(site_id)
        return SPSite(**d) if d else None

    def upsert(self, site: SPSite):
        if site.id not in self._sites.get("sites", {}):
            self._sites["sites"][site.id] = {}
        self._sites["sites"][site.id] = asdict(site)
        if not self._sites.get("active_id"):
            self._sites["active_id"] = site.id
        self._save_sites()

    def delete(self, site_id: str):
        self._sites.get("sites", {}).pop(site_id, None)
        if self._sites.get("active_id") == site_id:
            self._sites["active_id"] = next(iter(self._sites.get("sites", {})), "")
        self._save_sites()

    def set_active(self, site_id: str):
        if site_id in self._sites.get("sites", {}):
            self._sites["active_id"] = site_id
            self._save_sites()

    def active_site_id(self) -> str:
        return self._sites.get("active_id", "")

    # ---------- index state ----------
    def mark_index_run(self, site_id: str):
        self._state.setdefault(site_id, {})["last_full_index_ts"] = int(time.time())
        self._save_state()

    def get_site_state(self, site_id: str) -> Dict[str, Any]:
        return self._state.get(site_id, {})

    def update_item_state(self, site_id: str, item_id: str, etag: str, last_modified: str):
        s = self._state.setdefault(site_id, {})
        items = s.setdefault("items", {})
        items[item_id] = {"etag": etag, "lastModifiedDateTime": last_modified}
        self._save_state()

    def get_item_state(self, site_id: str, item_id: str) -> Dict[str, Any]:
        return self._state.get(site_id, {}).get("items", {}).get(item_id, {})
