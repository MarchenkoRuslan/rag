"""LLM answer generation with strict context-only prompt."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, cast

import httpx
from openai import OpenAI

from app.config import LLMProvider, Settings
from app.services.retrieval import RetrievedChunk
from app.utils.logging import get_logger

log = get_logger("generation")

SYSTEM_PROMPT = (
    "You are a precise assistant for a retrieval-augmented knowledge system.\n"
    "Answer ONLY based on the provided context.\n"
    "If the answer is not contained in the context, respond exactly with: "
    "I don't have enough information to answer this question.\n"
    "Use inline citations like [1], [2] that refer to the numbered context blocks.\n"
    "Do not invent facts or sources."
)

_MAX_RETRIES = 3


def build_openai_client(settings: Settings) -> OpenAI:
    return OpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout_seconds,
        max_retries=_MAX_RETRIES,
    )


def build_ollama_client(settings: Settings) -> httpx.Client:
    transport = httpx.HTTPTransport(retries=_MAX_RETRIES)
    return httpx.Client(
        timeout=settings.openai_timeout_seconds,
        transport=transport,
    )


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    lines = [
        "Context:",
    ]
    for c in chunks:
        lines.append(f"[{c.citation_id}] (source: {c.filename}, chunk {c.chunk_index}): {c.text}")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("Provide a concise answer with citations [N] where applicable.")
    return "\n".join(lines)


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    settings: Settings,
    *,
    openai_client: OpenAI | None = None,
    ollama_client: httpx.Client | None = None,
) -> str:
    if not chunks:
        return "I don't have enough information to answer this question."

    user_content = build_user_prompt(question, chunks)

    if settings.llm_provider == LLMProvider.OPENAI:
        owns_client = openai_client is None
        client = openai_client or build_openai_client(settings)
        try:
            resp = client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
            )
            text = (resp.choices[0].message.content or "").strip()
            log.info("llm_openai_ok", model=settings.llm_model)
            return text
        finally:
            if owns_client and hasattr(client, "close"):
                client.close()

    return _ollama_chat(user_content, settings, ollama_client)


def _ollama_chat(
    user_content: str,
    settings: Settings,
    client: httpx.Client | None = None,
) -> str:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
    }
    if client is None:
        http_client = build_ollama_client(settings)
        close_after = True
    else:
        http_client = client
        close_after = False
    try:
        r = http_client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message") or {}
        text = (msg.get("content") or "").strip()
        log.info("llm_ollama_ok", model=settings.llm_model)
        return text
    except Exception as e:
        log.exception("llm_ollama_error", error=str(e))
        raise
    finally:
        if close_after:
            http_client.close()


def generate_answer_stream(
    question: str,
    chunks: list[RetrievedChunk],
    settings: Settings,
    *,
    openai_client: OpenAI | None = None,
    ollama_client: httpx.Client | None = None,
) -> Iterator[str]:
    """Yield text tokens as they arrive from the LLM."""
    if not chunks:
        yield "I don't have enough information to answer this question."
        return

    user_content = build_user_prompt(question, chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    if settings.llm_provider == LLMProvider.OPENAI:
        owns_client = openai_client is None
        client = openai_client or build_openai_client(settings)
        try:
            stream = client.chat.completions.create(
                model=settings.llm_model,
                messages=cast(Any, messages),
                temperature=0.2,
                stream=True,
            )
            for raw in stream:
                chunk = cast(Any, raw)
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
            return
        except Exception as e:
            log.exception("llm_openai_stream_error", error=str(e))
            raise
        finally:
            if owns_client and hasattr(client, "close"):
                client.close()

    yield from _ollama_chat_stream(user_content, settings, ollama_client)


def _ollama_chat_stream(
    user_content: str,
    settings: Settings,
    client: httpx.Client | None = None,
) -> Iterator[str]:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": True,
    }
    if client is None:
        http_client = build_ollama_client(settings)
        close_after = True
    else:
        http_client = client
        close_after = False
    try:
        with http_client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("ollama_stream_bad_json", line_preview=line[:200])
                    continue
                msg = data.get("message") or {}
                token = msg.get("content", "")
                if token:
                    yield token
    except Exception as e:
        log.exception("llm_ollama_stream_error", error=str(e))
        raise
    finally:
        if close_after:
            http_client.close()
