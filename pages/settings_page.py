# pages/settings_page.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List
import streamlit as st

from modules.app_helpers import (
    load_static_map, save_static_map, list_files
)
from modules.vectorstore import ingest_sharepoint
from modules.url_store import load_sources, save_sources, add_source, update_source, delete_source, SPSite


def _ensure_dirs(TEMPLATE_DIR: Path, STATIC_DIR: Path):
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)


def _site_card(site: SPSite, idx: int):
    with st.container(border=True):
        st.markdown(f"**{site.name}**\n\n`{site.site_url}`")
        col1, col2 = st.columns([3,1])
        with col1:
            new_folder = st.text_input("Folder path (optional)", value=site.folder_path or "", key=f"fp_{idx}")
            new_notes = st.text_area("Notes", value=site.notes or "", key=f"nt_{idx}")
        with col2:
            st.caption(f"Last ingested: {site.last_ingested or '—'}")
            if st.button("Save", key=f"save_{idx}"):
                site.folder_path = new_folder.strip()
                site.notes = new_notes.strip()
                st.session_state["_save_site"] = site
            if st.button("Delete", key=f"del_{idx}"):
                st.session_state["_delete_site_id"] = site.id


def render_settings(cfg, *, BASE_DIR: Path, TEMPLATE_DIR: Path, STATIC_DIR: Path, STATIC_MAP_FILE: Path, STATIC_SECTIONS: List[str]):
    _ensure_dirs(TEMPLATE_DIR, STATIC_DIR)

    st.markdown("### ⚙️ Settings")
    st.markdown("<div class='sticky'><span class='badge'>Admin</span> Configure SharePoint, templates, and static library</div>", unsafe_allow_html=True)

    tabs = st.tabs(["SharePoint", "Templates", "Static Library", "Environment"])

    # ---------------- SharePoint ----------------
    with tabs[0]:
        st.subheader("SharePoint Sources")
        base_dir = str(BASE_DIR)
        sites = load_sources(base_dir)

        # add new
        with st.expander("Add SharePoint site", expanded=False):
            name = st.text_input("Display name", key="sp_add_name")
            url = st.text_input("Site URL (any page under the site is OK)", key="sp_add_url")
            folder = st.text_input("Root folder path (optional)", key="sp_add_folder", placeholder="Documents/Bid Management/2025")
            notes = st.text_area("Notes", key="sp_add_notes")
            if st.button("Add site"):
                if not (name.strip() and url.strip()):
                    st.warning("Please provide a display name and a site URL.")
                else:
                    it = add_source(base_dir, name=name, site_url=url, folder_path=folder, notes=notes)
                    st.success(f"Added: {it.name}")
                    st.rerun()

        if not sites:
            st.info("No SharePoint sources yet. Add one above.")
        else:
            selected_ids: List[str] = []
            for i, s in enumerate(sites):
                _site_card(s, i)
                if st.checkbox("Select for indexing", key=f"sel_{i}"):
                    selected_ids.append(s.id)
                st.markdown("---")

            # side effects from cards
            if st.session_state.get("_save_site"):
                site_upd: SPSite = st.session_state.pop("_save_site")
                update_source(base_dir, site_upd); save_sources(base_dir, sites)
                st.success("Saved changes.")
                st.rerun()
            if st.session_state.get("_delete_site_id"):
                sid = st.session_state.pop("_delete_site_id")
                delete_source(base_dir, sid)
                st.success("Deleted.")
                st.rerun()

            st.markdown("#### Indexing")
            colI1, colI2 = st.columns([1,2])
            with colI1:
                k = st.slider("Context Top-K (ingest evidence depth)", 3, 12, 6)
            with colI2:
                exts = st.multiselect("File types", ["pdf","docx","pptx","xlsx","txt"], default=["pdf","docx","pptx"])

            if st.button("⬇️ Pull & Index selected sites"):
                if not selected_ids:
                    st.warning("Select at least one site above.")
                else:
                    from modules.sharepoint import ingest_sharepoint_files
                    from modules.vectorstore import init_sharepoint_store, ingest_files
                    vs = init_sharepoint_store(cfg)
                    total = 0
                    with st.spinner("Downloading & indexing from SharePoint…"):
                        for s in sites:
                            if s.id not in selected_ids:
                                continue
                            try:
                                files = ingest_sharepoint_files(cfg, include_exts=exts, include_folder_ids=None)
                            except TypeError:
                                # fallback to helper in vectorstore if older signature
                                files = ingest_sharepoint(cfg)
                            for p in files:
                                try:
                                    ingest_files([p], vs, getattr(cfg, "CHUNK_SIZE", 1000))
                                    total += 1
                                except Exception:
                                    continue
                    st.success(f"Indexed {total} files across selected sites.")

    # ---------------- Templates ----------------
    with tabs[1]:
        st.subheader("Proposal Templates (.docx)")
        tmpl_files = [f for f in os.listdir(TEMPLATE_DIR) if f.lower().endswith(".docx")]
        st.write(f"Folder: `{TEMPLATE_DIR}`")
        up = st.file_uploader("Upload .docx templates", type=["docx"], accept_multiple_files=True, key="up_tpl")
        if st.button("Upload templates", disabled=not up):
            saved = 0
            for f in up or []:
                path = TEMPLATE_DIR / f.name
                with open(path, "wb") as fh:
                    fh.write(f.getvalue())
                saved += 1
            st.success(f"Uploaded {saved} templates.")
            st.rerun()

        if tmpl_files:
            del_name = st.selectbox("Delete a template", options=["(choose)"] + tmpl_files)
            if st.button("Delete template", disabled=(del_name=="(choose)")):
                try:
                    (TEMPLATE_DIR / del_name).unlink()
                    st.success("Deleted.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")
        else:
            st.info("No templates uploaded yet.")

    # ---------------- Static Library ----------------
    with tabs[2]:
        st.subheader("Static Sections Library")
        st.caption("Upload reusable .docx snippets and map them to section names.")
        st.write(f"Folder: `{STATIC_DIR}`")
        up2 = st.file_uploader("Upload static .docx files", type=["docx"], accept_multiple_files=True, key="up_static")
        if st.button("Upload static docs", disabled=not up2):
            saved = 0
            for f in up2 or []:
                path = STATIC_DIR / f.name
                with open(path, "wb") as fh:
                    fh.write(f.getvalue())
                saved += 1
            st.success(f"Uploaded {saved} static docs.")
            st.rerun()

        available = list_files(STATIC_DIR, ("docx",))
        mapping = load_static_map(STATIC_MAP_FILE)
        st.markdown("#### Map sections → files")
        new_map: Dict[str, str] = {**mapping}
        for sec in STATIC_SECTIONS:
            current = mapping.get(sec, "(none)")
            options = ["(none)"] + available
            try:
                idx = options.index(current)
            except ValueError:
                idx = 0
            choice = st.selectbox(sec, options=options, index=idx, key=f"map_{sec}")
            if choice == "(none)":
                new_map.pop(sec, None)
            else:
                new_map[sec] = choice
        if st.button("Save mapping"):
            save_static_map(STATIC_MAP_FILE, new_map)
            st.success("Mapping saved.")

    # ---------------- Environment ----------------
    with tabs[3]:
        st.subheader("Environment Variables")
        rows = [
            ("OPENAI_API_KEY", bool(os.getenv("OPENAI_API_KEY"))),
            ("SP_CLIENT_ID",   bool(os.getenv("SP_CLIENT_ID"))),
            ("SP_CLIENT_SECRET", bool(os.getenv("SP_CLIENT_SECRET"))),
            ("SP_TENANT_ID",   bool(os.getenv("SP_TENANT_ID"))),
            ("SP_SITE_URL",    bool(os.getenv("SP_SITE_URL"))),
        ]
        for k, ok in rows:
            st.write(f"**{k}**: {'✅ set' if ok else '❌ missing'}")
        st.caption("Set these in Azure App Service → Configuration, or in your local .env when developing.")

