"""Minimal Streamlit client for the RAG API."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping
from urllib.parse import quote

import httpx
import streamlit as st

DEFAULT_API = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")


def _render_sources(items: list[Mapping[str, Any]]) -> None:
    with st.expander("Sources"):
        for s in items:
            st.markdown(
                f"**[{s.get('citation_id')}]** `{s.get('filename')}` "
                f"chunk **{s.get('chunk_index')}** "
                f"(score {float(s.get('relevance_score', 0)):.3f})\n\n"
                f"{s.get('text', '')}"
            )


def _metrics_caption(metrics_data: Mapping[str, Any]) -> str:
    return (
        f"Response: **{float(metrics_data.get('response_time_ms', 0)):.1f} ms** | "
        f"retrieval: **{float(metrics_data.get('retrieval_time_ms', 0)):.1f} ms** | "
        f"LLM: **{float(metrics_data.get('generation_time_ms', 0)):.1f} ms** | "
        f"sources: **{metrics_data.get('num_sources_used', 0)}** | "
        f"hint: **{metrics_data.get('retrieval_accuracy_hint')}**"
    )


def _handle_chat_turn(user_prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
    with st.chat_message("assistant"):
        try:
            with httpx.Client(timeout=300.0) as query_http:
                qr = query_http.post(
                    f"{DEFAULT_API}/query",
                    json={"question": user_prompt},
                )
            qr.raise_for_status()
            body = qr.json()
            answer = body.get("answer", "")
            resp_sources = body.get("sources", [])
            resp_metrics = body.get("metrics", {})
            st.markdown(answer)
            if resp_sources:
                _render_sources(resp_sources)
            st.caption(_metrics_caption(resp_metrics))
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": resp_sources,
                    "metrics": resp_metrics,
                }
            )
        except (httpx.HTTPError, json.JSONDecodeError, TypeError) as exc:
            message = f"Request error: {exc}"
            st.error(message)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": message,
                    "sources": [],
                    "metrics": {},
                }
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
            with httpx.Client(timeout=300.0) as ingest_http:
                r = ingest_http.post(f"{DEFAULT_API}/ingest", files=files)
            r.raise_for_status()
            data = r.json()
            st.success(
                f"Done: {data.get('chunks_added', 0)} chunks from "
                f"{data.get('characters_extracted', 0)} characters"
            )
        except (httpx.HTTPError, json.JSONDecodeError, TypeError) as e:
            st.error(f"Ingest error: {e}")

    st.divider()
    st.subheader("Indexed documents")
    if st.button("Refresh list"):
        st.session_state.pop("docs_cache", None)
    try:
        with httpx.Client(timeout=30.0) as list_http:
            dr = list_http.get(f"{DEFAULT_API}/documents")
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
                        except httpx.HTTPError as ex:
                            st.error(f"Delete failed: {ex}")
    except (httpx.HTTPError, json.JSONDecodeError, TypeError) as e:
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
    _handle_chat_turn(prompt)
