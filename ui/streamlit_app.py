"""Minimal Streamlit client for the RAG API."""

from __future__ import annotations

import os
from typing import Any, Mapping
from urllib.parse import quote

import httpx
import streamlit as st

DEFAULT_API = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")


def _render_sources(sources: list[Mapping[str, Any]]) -> None:
    with st.expander("Sources"):
        for s in sources:
            st.markdown(
                f"**[{s.get('citation_id')}]** `{s.get('filename')}` "
                f"chunk **{s.get('chunk_index')}** "
                f"(score {float(s.get('relevance_score', 0)):.3f})\n\n"
                f"{s.get('text', '')}"
            )


def _metrics_caption(metrics: Mapping[str, Any]) -> str:
    return (
        f"Response: **{float(metrics.get('response_time_ms', 0)):.1f} ms** | "
        f"retrieval: **{float(metrics.get('retrieval_time_ms', 0)):.1f} ms** | "
        f"LLM: **{float(metrics.get('generation_time_ms', 0)):.1f} ms** | "
        f"sources: **{metrics.get('num_sources_used', 0)}** | "
        f"hint: **{metrics.get('retrieval_accuracy_hint')}**"
    )


st.set_page_config(page_title="RAG Knowledge System", layout="wide")
st.title("RAG Knowledge System")
st.caption("Search your documents with citations and latency metrics")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Document upload")
    st.text(f"API: {DEFAULT_API}")
    up = st.file_uploader("File (.txt or .pdf)", type=["txt", "pdf"])
    if up is not None and st.button("Ingest into index", type="primary"):
        try:
            files = {"file": (up.name, up.getvalue())}
            with httpx.Client(timeout=300.0) as client:
                r = client.post(f"{DEFAULT_API}/ingest", files=files)
            r.raise_for_status()
            data = r.json()
            st.success(
                f"Done: {data.get('chunks_added', 0)} chunks from "
                f"{data.get('characters_extracted', 0)} characters"
            )
        except Exception as e:
            st.error(f"Ingest error: {e}")

    st.divider()
    st.subheader("Indexed documents")
    if st.button("Refresh list"):
        st.session_state.pop("docs_cache", None)
    try:
        with httpx.Client(timeout=30.0) as client:
            dr = client.get(f"{DEFAULT_API}/documents")
        dr.raise_for_status()
        docs = dr.json().get("documents", [])
        if not docs:
            st.info("No documents yet")
        else:
            for d in docs:
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.write(
                        f"**{d['filename']}** — {d['chunk_count']} chunks"
                    )
                with c2:
                    key = f"delete_{d['filename']}"
                    if st.button("Remove", key=key):
                        try:
                            enc = quote(d["filename"], safe="")
                            with httpx.Client(timeout=60.0) as del_client:
                                rr = del_client.delete(
                                    f"{DEFAULT_API}/documents/{enc}"
                                )
                            rr.raise_for_status()
                            st.success("Removed")
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Delete failed: {ex}")
    except Exception as e:
        st.warning(f"Could not fetch /documents: {e}")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("sources"):
            _render_sources(m["sources"])
            if m.get("metrics"):
                st.caption(_metrics_caption(m["metrics"]))

prompt = st.chat_input("Ask a question about your documents...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with httpx.Client(timeout=300.0) as client:
                qr = client.post(
                    f"{DEFAULT_API}/query",
                    json={"question": prompt},
                )
            qr.raise_for_status()
            body = qr.json()
            answer = body.get("answer", "")
            sources = body.get("sources", [])
            metrics = body.get("metrics", {})
            st.markdown(answer)
            if sources:
                _render_sources(sources)
            st.caption(_metrics_caption(metrics))
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "metrics": metrics,
                }
            )
        except Exception as e:
            err = f"Request error: {e}"
            st.error(err)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": err,
                    "sources": [],
                    "metrics": {},
                }
            )
