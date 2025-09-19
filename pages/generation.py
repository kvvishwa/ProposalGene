# pages/generation.py
from __future__ import annotations
import os, json, hashlib
from time import strftime, localtime
import streamlit as st

from modules.app_helpers import (
    load_static_map, list_blueprints, save_blueprint, load_blueprint,
    sp_index_stats, st_rerun_compat
)
from modules.proposal_builder import BuildOptions, build_proposal
from modules import llm_processing


# ----------------------------
# Session-state bootstrapping
# ----------------------------
def _init_gen_state():
    ss = st.session_state
    ss.setdefault("gen_tpl", "— choose a template —")
    ss.setdefault("gen_template", "")          # compat alias (if referenced elsewhere)
    ss.setdefault("gen_add_headings", True)    # fixes earlier AttributeError
    ss.setdefault("gen_include_vp", True)
    ss.setdefault("gen_dyn_sel", [])           # dynamic sections (names) selected
    ss.setdefault("gen_dyn_weights", {})       # section -> weight
    ss.setdefault("gen_static_sel", [])        # selected static sections
    ss.setdefault("gen_extra_notes", "")
    # builder knobs
    ss.setdefault("gen_use_anchors", True)
    ss.setdefault("gen_include_sources", True)
    ss.setdefault("gen_top_k", 6)
    ss.setdefault("gen_rec_style", "paragraphs")
    ss.setdefault("gen_per_section_k", 25)
    ss.setdefault("gen_page_breaks", True)
    ss.setdefault("gen_add_toc", True)
    ss.setdefault("gen_tpl_has_headings", True)
    # outputs
    ss.setdefault("out_draft_bytes", b"")
    ss.setdefault("out_recs_bytes", b"")
    ss.setdefault("last_generation_meta", {})
    ss.setdefault("dyn_recos_preview", {})


# ----------------------------
# Understanding → plan/blueprint
# ----------------------------
def _hash_facts(facts: dict) -> str:
    try:
        s = json.dumps(facts or {}, sort_keys=True, ensure_ascii=False)
    except Exception:
        s = str(facts or "")
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _compute_dynamic_recos_and_blueprint(facts, *, up_store=None, sp_store=None, oai=None, cfg=None):
    plan = None

    # Try rich signatures first; fall back gracefully
    for fn in (
        lambda: llm_processing.plan_dynamic_sections(facts=facts, up_store=up_store, sp_store=sp_store, oai=oai, cfg=cfg),
        lambda: llm_processing.plan_dynamic_sections(facts=facts, stores={"uploaded": up_store, "sharepoint": sp_store}, oai=oai, cfg=cfg),
        lambda: llm_processing.plan_dynamic_sections(facts),
    ):
        try:
            plan = fn()
            if plan:
                break
        except Exception:
            continue

    if not plan:
        return {"plan": None, "recs": [], "blueprint": None}

    # Normalize recommendations list
    recs = plan.get("recommendations") or plan.get("sections") or []

    # Build a blueprint from the plan (fallback-safe)
    blueprint = None
    for fn in (
        lambda: llm_processing.plan_to_blueprint(plan, cfg=cfg),
        lambda: llm_processing.plan_to_blueprint(plan),
    ):
        try:
            blueprint = fn()
            if blueprint:
                break
        except Exception:
            continue

    return {"plan": plan, "recs": recs, "blueprint": blueprint}


def _rec_name(rec) -> str:
    if isinstance(rec, dict):
        return rec.get("name") or rec.get("section") or rec.get("title") or ""
    return str(rec or "").strip()


def _rec_weight(rec) -> int:
    if isinstance(rec, dict):
        for k in ("weight", "priority", "score"):
            v = rec.get(k)
            if isinstance(v, (int, float)):
                return int(v)
    return 0


# One-time eager init for the Generation tab
def ensure_gen_tab_initialized():
    ss = st.session_state
    ss.setdefault("gen_initialized", False)
    ss.setdefault("rfp_facts", {})
    ss.setdefault("understanding_plan", None)
    ss.setdefault("gen_dynamic_recos", [])
    ss.setdefault("gen_blueprint", None)
    ss.setdefault("gen_last_facts_hash", "")

    facts = ss.get("rfp_facts") or {}
    curr_hash = _hash_facts(facts)

    need_init = (not ss["gen_initialized"]) or (ss["gen_last_facts_hash"] != curr_hash)
    if not need_init:
        return False

    # Pull stores/clients if you stash them in state
    up_store = ss.get("uploaded_store")
    sp_store = ss.get("sp_store")
    oai = ss.get("oai_client")
    cfg = ss.get("cfg")

    result = _compute_dynamic_recos_and_blueprint(
        facts, up_store=up_store, sp_store=sp_store, oai=oai, cfg=cfg
    )

    recs = result["recs"] or []
    rec_names = [n for n in (_rec_name(r) for r in recs) if n]
    weights = {n: _rec_weight(r) for (n, r) in (( _rec_name(r), r) for r in recs) if n}

    # Seed defaults that the UI actually uses
    ss["gen_dynamic_recos"] = recs
    ss["gen_blueprint"] = result["blueprint"]
    # order by weight desc (ties: input order)
    if rec_names:
        rec_names_sorted = sorted(rec_names, key=lambda n: weights.get(n, 0), reverse=True)
    else:
        rec_names_sorted = []
    ss["gen_dyn_sel"] = rec_names_sorted               # <- this is what the widget reads
    ss["gen_dyn_weights"] = weights

    ss["gen_last_facts_hash"] = curr_hash
    ss["gen_initialized"] = True

    # Re-render so the multiselect shows preselected defaults
    st_rerun_compat()
    return True


# ----------------------------
# Page renderer
# ----------------------------
def render_generation(cfg, oai, BASE_DIR, STATIC_DIR, TEMPLATE_DIR, STATIC_MAP_FILE, STATIC_SECTIONS):
    ss = st.session_state
    _init_gen_state()

    # Eager-load recommendations on first paint / when facts change
    ensure_gen_tab_initialized()

    if ss.get("pending_blueprint"):
        bp = ss.pending_blueprint or {}
        ss.gen_tpl = bp.get("template", ss.gen_tpl)
        if bp.get("static_sel") is not None:
            ss.gen_static_sel = bp.get("static_sel") or ss.gen_static_sel
        ss.gen_dyn_sel = bp.get("dyn_sel", ss.gen_dyn_sel)
        ss.gen_include_sources = bool(bp.get("include_sources", ss.gen_include_sources))
        ss.gen_top_k = int(bp.get("top_k", ss.gen_top_k))
        ss.gen_use_anchors = bool(bp.get("use_anchors", ss.gen_use_anchors))
        ss.gen_add_headings = bool(bp.get("add_headings", ss.gen_add_headings))
        ss.gen_rec_style = bp.get("rec_style", ss.gen_rec_style)
        ss.gen_per_section_k = int(bp.get("per_section_k", ss.gen_per_section_k))
        ss.gen_add_toc = bool(bp.get("add_toc", ss.gen_add_toc))
        ss.gen_page_breaks = bool(bp.get("page_breaks", ss.gen_page_breaks))
        ss.gen_tpl_has_headings = bool(bp.get("template_has_headings", ss.gen_tpl_has_headings))
        ss.pending_blueprint = None

    st.markdown("### 🧩 Proposal Generation")
    st.markdown(
        '<div class="sticky"><span class="badge">Step 2</span> Single DOCX from Template + Static + Dynamic (SharePoint-grounded)</div>',
        unsafe_allow_html=True
    )

    with st.expander("SharePoint index status", expanded=False):
        chunks, vec_path = sp_index_stats(cfg)
        st.write(f"Index path: `{vec_path}` • Chunks: **{chunks}**")
        if chunks == 0:
            st.warning("SharePoint index is empty. Dynamic recommendations will be generic.")

    templates = sorted([f for f in os.listdir(TEMPLATE_DIR) if f.lower().endswith(".docx")])
    tpl_options = ["— choose a template —"] + templates
    try:
        tpl_index = tpl_options.index(ss.get("gen_tpl", "— choose a template —"))
    except ValueError:
        tpl_index = 0
    _ = st.selectbox(
        "Template", options=tpl_options, index=tpl_index, key="gen_tpl",
        help="Upload templates under Settings → Static Library → Proposal Templates."
    )

    static_map = load_static_map(STATIC_MAP_FILE)
    available_static = [sec for sec in STATIC_SECTIONS if static_map.get(sec)]
    if not ss.get("gen_static_sel"):
        ss.gen_static_sel = available_static

    st.markdown("#### Static sections (from your mapped .docx)")
    _ = st.multiselect(
        "Select static sections", options=available_static,
        default=ss.get("gen_static_sel", available_static), key="gen_static_sel"
    )

    # Manual refresh (recompute from Understanding)
    
    # ----- Dynamic sections -----
    st.markdown("#### Dynamic sections (SharePoint-grounded recommendations)")
    # Recommended names from plan
    plan_recs = ss.get("gen_dynamic_recos", [])
    rec_names = [n for n in (_rec_name(r) for r in plan_recs) if n]

    # Curated pool (acts as friendly fallback)
    curated = [
        "Executive Summary","Approach & Methodology","Staffing Plan & Roles","Transition & Knowledge Transfer",
        "Service Levels & Governance","Assumptions & Exclusions","Risk & Mitigation Plan","Project Governance",
        "Change Management","Quality Management","Compliance & Security","Timeline & Milestones"
    ]

    # Final options = union(recommended, curated) minus anything covered by static
    blocked = set(ss.gen_static_sel or [])
    dynamic_options = sorted(set(curated + rec_names + (ss.get("gen_dyn_sel") or [])) - set(ss.get("gen_static_sel") or []))

    if st.button("🔄 Refresh recommendations from Understanding"):
        ss["gen_initialized"] = False
        # recompute & rerender
        ensure_gen_tab_initialized()
        st.success("Recommendations refreshed from Understanding.")


    # Defaults = preselected recommendations (already sorted by weight in init)
    default_dyn = [n for n in ss.get("gen_dyn_sel", rec_names) if n in dynamic_options]
    _ = st.multiselect(
        "Select dynamic sections",
        options=dynamic_options,
        default=default_dyn,
        key="gen_dyn_sel"
    )

    # ----- Final order -----
    st.markdown("#### Final order")
    combined_options = (ss.gen_static_sel or []) + [f"[Dyn] {x}" for x in (ss.gen_dyn_sel or [])]
    final_order_widget = st.multiselect(
        "Pick the final sequence (drag to re-order)",
        options=combined_options,
        default=combined_options,
        key="gen_final_order_widget"
    )
    final_order = final_order_widget

    # Builder knobs
    _ = st.checkbox("Place at template anchors (SDT/Bookmark/Marker)", value=ss.get("gen_use_anchors", True), key="gen_use_anchors")
    _ = st.checkbox("Append 'Sources' line for dynamic sections", value=ss.get("gen_include_sources", True), key="gen_include_sources")
    _ = st.slider("Context Top-K (SharePoint)", min_value=3, max_value=12, value=int(ss.get("gen_top_k", 6)), step=1, key="gen_top_k")
    _ = st.selectbox("Recommendations style", options=["bullets","paragraphs"],
                     index=["bullets","paragraphs"].index(ss.get("gen_rec_style","paragraphs")), key="gen_rec_style")
    _ = st.number_input("Per-section Top-K override (0 = use slider)", min_value=0, max_value=50,
                        value=int(ss.get("gen_per_section_k", 25)), step=1, key="gen_per_section_k")
    _ = st.checkbox("Page break between sections", value=ss.get("gen_page_breaks", True), key="gen_page_breaks")
    _ = st.checkbox("Insert Table of Contents at top", value=ss.get("gen_add_toc", True), key="gen_add_toc")
    _ = st.checkbox("Template has headings at anchors (don’t add heading in dynamic content)", value=ss.get("gen_tpl_has_headings", True), key="gen_tpl_has_headings")

    # Blueprints
    st.markdown("#### Blueprints")
    bp_cols = st.columns([2,1,1,2])
    with bp_cols[0]:
        bp_name = st.text_input("Name a blueprint to save current setup", value="")
    with bp_cols[1]:
        if st.button("Save blueprint", use_container_width=True, disabled=(bp_name.strip()=="")):
            payload = {
                "template": ss.gen_tpl, "static_sel": ss.gen_static_sel, "dyn_sel": ss.gen_dyn_sel,
                "include_sources": bool(ss.gen_include_sources), "top_k": int(ss.gen_top_k), "tone": "Professional",
                "use_anchors": bool(ss.gen_use_anchors), "add_headings": bool(ss.gen_add_headings), "rec_style": ss.gen_rec_style,
                "per_section_k": int(ss.gen_per_section_k), "add_toc": bool(ss.gen_add_toc), "page_breaks": bool(ss.gen_page_breaks),
                "template_has_headings": bool(ss.gen_tpl_has_headings),
            }
            path = save_blueprint(BASE_DIR, bp_name.strip(), payload)
            st.success(f"Saved blueprint → {path}")
    with bp_cols[2]:
        choices = ["(choose)"] + list_blueprints(BASE_DIR)
        chosen = st.selectbox("Load blueprint", options=choices, index=0)
    with bp_cols[3]:
        if st.button("Apply blueprint", use_container_width=True, disabled=(chosen=="(choose)")):
            bp = load_blueprint(BASE_DIR, chosen)
            if bp:
                ss.pending_blueprint = bp
                st_rerun_compat()

    # Generate
    disabled = (ss.gen_tpl == "— choose a template —") or (len(final_order) == 0)
    if st.button("Generate Draft + Recommendations", disabled=disabled, use_container_width=True):
        if ss.gen_tpl == "— choose a template —":
            st.warning("Please choose a template.")
        elif len(final_order) == 0:
            st.warning("Choose at least one section in the final order.")
        else:
            ss.dyn_recos_preview = {}
            template_path = str((TEMPLATE_DIR / ss.gen_tpl).resolve())
            static_map_now = load_static_map(STATIC_MAP_FILE)
            static_paths = {
                sec: str((STATIC_DIR / static_map_now[sec]).resolve())
                for sec in ss.gen_static_sel if static_map_now.get(sec)
            }
            opts = BuildOptions(
                use_anchors=ss.gen_use_anchors, template_has_headings=ss.gen_tpl_has_headings, page_breaks=ss.gen_page_breaks,
                include_sources=ss.gen_include_sources, add_toc=ss.gen_add_toc, rec_style=ss.gen_rec_style,
                top_k_default=int(ss.gen_top_k), top_k_per_section=int(ss.gen_per_section_k), tone="Professional",
                static_heading_mode="demote", facts=ss.rfp_facts or {}
            )
            with st.spinner("Composing draft and recommendations…"):
                draft_bytes, recs_bytes, preview, meta = build_proposal(
                    template_path=template_path, static_paths=static_paths,
                    final_order=final_order, oai=oai, cfg=cfg, opts=opts
                )
            ss.out_draft_bytes = draft_bytes
            ss.out_recs_bytes = recs_bytes
            ss.dyn_recos_preview = preview
            meta["template"] = ss.gen_tpl
            meta["final_order"] = final_order
            meta["timestamp"] = strftime("%Y-%m-%d %H:%M:%S", localtime())
            ss.last_generation_meta = meta
            st.success("Draft generated. See the 'Latest output' panel below to download anytime.")

    # Latest output
    st.markdown("### Latest output")
    with st.container(border=True):
        meta = ss.last_generation_meta or {}
        if ss.out_draft_bytes:
            tpl_name = meta.get("template", "—")
            ts = meta.get("timestamp", "—")
            order_preview = ", ".join(meta.get("final_order", [])) or "—"
            st.caption(f"Generated: **{ts}** • Template: **{tpl_name}**")
            with st.expander("Show final sequence used", expanded=False):
                st.write(order_preview)
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "⬇ Download Draft DOCX",
                    data=ss.out_draft_bytes,
                    file_name="Draft_Response_Static+Dynamic.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
            with c2:
                if ss.out_recs_bytes:
                    st.download_button(
                        "⬇ Download Recommendations DOCX",
                        data=ss.out_recs_bytes,
                        file_name="Dynamic_Recommendations.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                    )

            if ss.dyn_recos_preview:
                st.markdown("#### Dynamic recommendations used (preview)")
                all_md = ["# Dynamic Recommendations (Preview)\n"]
                for sec_name, pkg in ss.dyn_recos_preview.items():
                    st.markdown(f"**{sec_name}**")
                    st.markdown(pkg["md"])
                    if pkg.get("evidence"):
                        with st.expander(f"Show retrieval diagnostics — {sec_name}", expanded=False):
                            for i, ev in enumerate(pkg["evidence"], start=1):
                                src = ev.get("source") if isinstance(ev, dict) else getattr(ev, "source", "")
                                txt = ev.get("text") if isinstance(ev, dict) else getattr(ev, "text", "")
                                st.markdown(f"**[{i}]** `{src}`")
                                st.code(txt[:1200], language="text")
                    if pkg.get("sources"):
                        st.caption("Sources: " + ", ".join(pkg["sources"]))
                    all_md.append(f"## {sec_name}\n\n{pkg['md']}\n")
                    if pkg.get("sources"):
                        all_md.append(f"_Sources: {', '.join(pkg['sources'])}_\n")
                md_blob = "\n".join(all_md).encode("utf-8")
                st.download_button(
                    "⬇ Download recommendations (Markdown)",
                    data=md_blob,
                    file_name="dynamic_recommendations.md",
                    mime="text/markdown"
                )
        else:
            st.caption("No generated files yet. Click **Generate** above to create your draft.")
