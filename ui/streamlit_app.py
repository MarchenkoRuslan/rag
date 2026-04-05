"""Minimal Streamlit client for the RAG API."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

import httpx
import streamlit as st
from ui.helpers import _format_api_error, _req_headers, _safe_float

DEFAULT_API = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
_DEFAULT_TOP_K = 5
_DEFAULT_RELEVANCE = 0.25
_MAX_QUESTION_LENGTH = 4000


def _render_sources(items: list[Mapping[str, Any]]) -> None:
    with st.expander("Sources"):
        for s in items:
            st.markdown(
                f"**[{s.get('citation_id')}]** `{s.get('filename')}` "
                f"chunk **{s.get('chunk_index')}** "
                f"(score {_safe_float(s.get('relevance_score', 0)):.3f})\n\n"
                f"{s.get('text', '')}"
            )


def _metrics_caption(metrics_data: Mapping[str, Any]) -> str:
    r_ms = _safe_float(metrics_data.get("response_time_ms", 0))
    ret_ms = _safe_float(metrics_data.get("retrieval_time_ms", 0))
    gen_ms = _safe_float(metrics_data.get("generation_time_ms", 0))
    n_src = metrics_data.get("num_sources_used", 0)
    hint = metrics_data.get("retrieval_accuracy_hint")
    return (
        f"Response: **{r_ms:.1f} ms** | "
        f"retrieval: **{ret_ms:.1f} ms** | "
        f"LLM: **{gen_ms:.1f} ms** | "
        f"sources: **{n_src}** | "
        f"hint: **{hint}**"
    )


def _fetch_health_json() -> dict[str, Any] | None:
    try:
        with httpx.Client(timeout=15.0) as client:
            health_response = client.get(f"{DEFAULT_API}/health", headers=_req_headers())
        health_response.raise_for_status()
        body = health_response.json()
        return body if isinstance(body, dict) else None
    except (httpx.HTTPError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _handle_chat_turn(
    user_prompt: str,
    selected_top_k: int,
    selected_relevance_threshold: float,
) -> None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
    with st.chat_message("assistant"):
        try:
            payload = {
                "question": user_prompt,
                "top_k": selected_top_k,
                "relevance_threshold": selected_relevance_threshold,
            }
            with httpx.Client(timeout=300.0) as query_http:
                query_response = query_http.post(
                    f"{DEFAULT_API}/query",
                    json=payload,
                    headers=_req_headers(),
                )
            query_response.raise_for_status()
            query_body = query_response.json()
            answer = query_body.get("answer", "")
            resp_sources = query_body.get("sources", [])
            resp_metrics = query_body.get("metrics", {})
            st.markdown(answer)
            if query_body.get("index_empty"):
                st.info("The index is empty. Upload a document from the sidebar.")
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
            message = _format_api_error(exc)
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

health = _fetch_health_json()
if health is None:
    st.warning(f"Cannot reach API at {DEFAULT_API}. Check service availability and credentials.")
elif health.get("index_empty"):
    st.info("The index is empty. Upload a document from the sidebar to enable answers.")

with st.sidebar:
    st.subheader("Document upload")
    st.text(f"API: {DEFAULT_API}")
    top_k = st.number_input(
        "top_k (retrieval)",
        min_value=1,
        max_value=50,
        value=_DEFAULT_TOP_K,
        step=1,
    )
    relevance_threshold = st.number_input(
        "Relevance threshold",
        min_value=0.0,
        max_value=1.0,
        value=_DEFAULT_RELEVANCE,
        step=0.01,
        format="%.2f",
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    up = st.file_uploader("File (.txt or .pdf)", type=["txt", "pdf"])
    if up is not None and st.button("Ingest into index", type="primary"):
        try:
            files = {"file": (up.name, up.getvalue())}
            with httpx.Client(timeout=300.0) as ingest_http:
                response = ingest_http.post(
                    f"{DEFAULT_API}/ingest",
                    files=files,
                    headers=_req_headers(),
                )
            response.raise_for_status()
            data = response.json()
            st.success(
                f"Done: {data.get('chunks_added', 0)} chunks from "
                f"{data.get('characters_extracted', 0)} characters"
            )
        except (httpx.HTTPError, json.JSONDecodeError, TypeError) as err:
            st.error(_format_api_error(err))

    st.divider()
    st.subheader("Indexed documents")
    if st.button("Refresh list"):
        st.session_state.pop("docs_cache", None)
    try:
        with httpx.Client(timeout=30.0) as list_http:
            docs_response = list_http.get(f"{DEFAULT_API}/documents", headers=_req_headers())
        docs_response.raise_for_status()
        docs_body = docs_response.json()
        docs = docs_body.get("documents", []) if isinstance(docs_body, dict) else []
        if not isinstance(docs, list):
            docs = []
        if not docs:
            st.info("No documents yet")
        else:
            for d in docs:
                if not isinstance(d, Mapping):
                    continue
                filename = str(d.get("filename", "unknown"))
                chunk_count = int(_safe_float(d.get("chunk_count", 0), 0))
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.write(f"**{filename}** — {chunk_count} chunks")
                with c2:
                    key = f"delete_{filename}"
                    if st.button("Remove", key=key):
                        try:
                            enc = quote(filename, safe="")
                            with httpx.Client(timeout=60.0) as del_client:
                                rr = del_client.delete(
                                    f"{DEFAULT_API}/documents/{enc}",
                                    headers=_req_headers(),
                                )
                            rr.raise_for_status()
                            st.success("Removed")
                            st.rerun()
                        except httpx.HTTPError as ex:
                            st.error(_format_api_error(ex))
    except (httpx.HTTPError, json.JSONDecodeError, TypeError) as e:
        st.warning(_format_api_error(e))

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("sources"):
            _render_sources(m["sources"])
            if m.get("metrics"):
                st.caption(_metrics_caption(m["metrics"]))

prompt = st.chat_input("Ask a question about your documents...")
if prompt:
    cleaned = prompt.strip()
    if not cleaned:
        st.warning("Question must not be empty.")
        st.stop()
    if len(cleaned) > _MAX_QUESTION_LENGTH:
        st.warning(f"Question is too long (max {_MAX_QUESTION_LENGTH} characters).")
        st.stop()
    _handle_chat_turn(
        cleaned,
        int(top_k),
        float(relevance_threshold),
    )
