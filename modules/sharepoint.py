import os
import time
import tempfile
from urllib.parse import urlparse
import requests
import msal

# ---------- URL NORMALIZATION ----------
def canonicalize_site_url(input_url: str) -> str:
    """
    Accepts:
      - Clean site URL like: https://<tenant>.sharepoint.com/sites/<SiteName>
      - Deep page URL like: https://<tenant>.sharepoint.com/sites/<SiteName>/SitePages/Home.aspx
      - Even library/file URLs under the same site.
    Returns canonical site root:
      https://<tenant>.sharepoint.com/sites/<SiteName>
    """
    if not input_url:
        return input_url
    parsed = urlparse(input_url)
    parts = [p for p in parsed.path.split("/") if p]
    try:
        idx = parts.index("sites")
        site_name = parts[idx + 1] if idx + 1 < len(parts) else ""
        if site_name:
            return f"https://{parsed.netloc}/sites/{site_name}"
    except ValueError:
        pass
    return input_url

# ---------- AUTH ----------
def get_token(cfg):
    app = msal.ConfidentialClientApplication(
        client_id=cfg.SP_CLIENT_ID,
        authority=cfg.SP_AUTHORITY,
        client_credential=cfg.SP_CLIENT_SECRET,
    )
    token = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
    if "access_token" not in token:
        detail = token.get("error_description") or token.get("error") or "Auth failed"
        raise RuntimeError(f"SharePoint auth error: {detail}")
    return token["access_token"]

# ---------- LOW-LEVEL HELPERS (paging + retry) ----------
def _request_json(url: str, headers: dict, method: str = "GET", retries: int = 5, backoff: float = 0.8, timeout: int = 30):
    last_exc = None
    for attempt in range(retries):
        resp = requests.request(method, url, headers=headers, timeout=timeout)
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff * (2 ** attempt))
            continue
        try:
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            break
    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown SharePoint request error")

def _paged_items(url: str, headers: dict):
    while url:
        data = _request_json(url, headers)
        for it in data.get("value", []):
            yield it
        url = data.get("@odata.nextLink")

# ---------- SITE / DRIVES ----------
def _resolve_site_id(site_url: str, token: str) -> str:
    parsed = urlparse(site_url)
    ep = f"https://graph.microsoft.com/v1.0/sites/{parsed.netloc}:{parsed.path}"
    headers = {"Authorization": f"Bearer {token}"}
    data = _request_json(ep, headers)
    return data["id"]

def _list_site_drives(site_id: str, token: str):
    ep = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives"
    headers = {"Authorization": f"Bearer {token}"}
    return list(_paged_items(ep, headers))

def _list_children(drive_id: str, token: str, item_id: str | None):
    base = f"https://graph.microsoft.com/v1.0/drives/{drive_id}"
    ep = f"{base}/items/{item_id}/children" if item_id else f"{base}/root/children"
    headers = {"Authorization": f"Bearer {token}"}
    return list(_paged_items(ep, headers))

# ---------- PUBLIC: FOLDER DISCOVERY ----------
def list_all_folders(cfg):
    """
    Returns: list of {'driveId', 'id', 'path'} for EVERY folder in EVERY document library.
    """
    token = get_token(cfg)
    site_id = _resolve_site_id(canonicalize_site_url(cfg.SP_SITE_URL), token)
    drives = _list_site_drives(site_id, token)
    headers = {"Authorization": f"Bearer {token}"}

    folders = []
    for d in drives:
        drive_id = d["id"]
        drive_name = d.get("name") or "Documents"

        # seed with root folders
        stack = []
        for ch in _list_children(drive_id, token, None):
            if ch.get("folder"):
                stack.append((ch["id"], f"{drive_name}/{ch['name']}"))

        # depth-first traversal
        while stack:
            fid, fpath = stack.pop()
            folders.append({"driveId": drive_id, "id": fid, "path": fpath})
            url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{fid}/children"
            for ch in _paged_items(url, headers):
                if ch.get("folder"):
                    stack.append((ch["id"], f"{fpath}/{ch['name']}"))

    folders.sort(key=lambda x: x["path"].lower())
    return folders

# ---------- PUBLIC: FILE FETCH (RECURSIVE) ----------
def _download_file(drive_id: str, item_id: str, name: str, token: str) -> str:
    dl = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item_id}/content"
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(dl, headers=headers, stream=True)
    r.raise_for_status()
    suffix = os.path.splitext(name)[1] or ".bin"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 15):
            if chunk:
                f.write(chunk)
    return tmp

def _walk_and_collect_files(drive_id: str, token: str, start_item_id: str | None, include_exts: set[str] | None):
    headers = {"Authorization": f"Bearer {token}"}
    files = []

    def _process_children(folder_id: str):
        url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{folder_id}/children"
        for ch in _paged_items(url, headers):
            if ch.get("folder"):
                stack.append(ch["id"])
            elif ch.get("file"):
                name = ch.get("name", "")
                _, ext = os.path.splitext(name)
                ext = ext.lower().lstrip(".")
                if include_exts and ext not in include_exts:
                    continue
                files.append(_download_file(drive_id, ch["id"], name, token))

    # init stack
    stack = []
    if start_item_id:
        stack.append(start_item_id)
    else:
        # include any root files, too
        for ch in _list_children(drive_id, token, None):
            if ch.get("folder"):
                stack.append(ch["id"])
            elif ch.get("file"):
                name = ch.get("name", "")
                _, ext = os.path.splitext(name)
                ext = ext.lower().lstrip(".")
                if include_exts and ext not in include_exts:
                    continue
                files.append(_download_file(drive_id, ch["id"], name, token))

    while stack:
        _process_children(stack.pop())

    return files

def fetch_files_from_sharepoint(cfg, include_exts=None, include_folder_ids=None):
    """
    Recursively fetch files from the entire site (all drives),
    or limit to folders listed in include_folder_ids (driveItem IDs).
    include_exts: iterable of extensions (e.g., ['pdf','docx','pptx'])
    include_folder_ids: list of folder item IDs (strings) across any drive
    Returns list of local temp file paths.
    """
    include_exts = {ext.lower().lstrip(".") for ext in include_exts} if include_exts else None

    token = get_token(cfg)
    site_id = _resolve_site_id(canonicalize_site_url(cfg.SP_SITE_URL), token)
    drives = _list_site_drives(site_id, token)

    all_local_files = []
    if include_folder_ids:
        # Try each selected folder id across all drives (non-members will 404; just skip)
        for d in drives:
            for fid in include_folder_ids:
                try:
                    all_local_files.extend(_walk_and_collect_files(d["id"], token, start_item_id=fid, include_exts=include_exts))
                except requests.HTTPError:
                    continue
    else:
        # Walk every drive from root
        for d in drives:
            all_local_files.extend(_walk_and_collect_files(d["id"], token, start_item_id=None, include_exts=include_exts))

    return all_local_files

def ingest_sharepoint_files(cfg, include_exts=None, include_folder_ids=None):
    """Download files from SharePoint and return a list of local file paths."""
    return fetch_files_from_sharepoint(cfg, include_exts=include_exts, include_folder_ids=include_folder_ids)
