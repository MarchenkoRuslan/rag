"""LLM answer generation with strict context-only prompt."""

from __future__ import annotations

import httpx
from openai import OpenAI

from app.config import LLMProvider, Settings
from app.services.retrieval import RetrievedChunk
from app.utils.logging import get_logger

log = get_logger("generation")

SYSTEM_PROMPT = """You are a precise assistant for a retrieval-augmented knowledge system.
Answer ONLY based on the provided context.
If the answer is not contained in the context, respond exactly with: I don't have enough information to answer this question.
Use inline citations like [1], [2] that refer to the numbered context blocks.
Do not invent facts or sources."""


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    lines = [
        "Context:",
    ]
    for c in chunks:
        lines.append(
            f"[{c.citation_id}] (source: {c.filename}, chunk {c.chunk_index}): {c.text}"
        )
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("Provide a concise answer with citations [N] where applicable.")
    return "\n".join(lines)


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    settings: Settings,
) -> str:
    if not chunks:
        return "I don't have enough information to answer this question."

    user_content = build_user_prompt(question, chunks)

    if settings.llm_provider == LLMProvider.OPENAI:
        client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
        )
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

    return _ollama_chat(user_content, settings)


def _ollama_chat(user_content: str, settings: Settings) -> str:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
    }
    timeout = settings.openai_timeout_seconds
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        msg = data.get("message") or {}
        text = (msg.get("content") or "").strip()
        log.info("llm_ollama_ok", model=settings.llm_model)
        return text
    except Exception as e:
        log.exception("llm_ollama_error", error=str(e))
        raise
