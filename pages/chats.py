
# pages/chats.py
from __future__ import annotations
import streamlit as st
from openai import OpenAI

from modules.sp_retrieval import get_sp_evidence_for_question
from modules.vectorstore import init_sharepoint_store


def render_sharepoint_chat(cfg, oai: OpenAI):
    st.header("💬 SharePoint Chat")
    st.caption("Ask questions across all indexed SharePoint sources.")

    msgs = st.session_state.get("sp_chat_messages", [])
    for m in msgs:
        st.chat_message(m["role"]).write(m["content"])

    q = st.chat_input("Ask about SharePoint docs…")
    if q:
        msgs.append({"role": "user", "content": q})
        st.chat_message("user").write(q)

        evs = get_sp_evidence_for_question(q, st.session_state.get("rfp_facts"), cfg, k=6)
        context = "\n\n".join(e.text for e in evs)

        prompt = f"Context from SharePoint:\n{context}\n\nQuestion: {q}\nAnswer clearly."
        res = oai.chat.completions.create(
            model=cfg.ANALYSIS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        ans = res.choices[0].message.content.strip()

        msgs.append({"role": "assistant", "content": ans})
        st.chat_message("assistant").write(ans)
        st.session_state["sp_chat_messages"] = msgs
