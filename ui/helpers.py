"""HTTP/parsing helpers for the Streamlit UI (no Streamlit import)."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import httpx


def _req_headers() -> dict[str, str]:
    headers = {"X-Request-ID": str(uuid.uuid4())}
    api_key = os.environ.get("RAG_API_KEY", "").strip()
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _format_validation_detail(detail: list[Any]) -> list[str]:
    parts: list[str] = []
    for item in detail:
        if isinstance(item, dict):
            loc = item.get("loc")
            msg = item.get("msg")
            if not msg:
                continue
            if isinstance(loc, list):
                parts.append(f"{'.'.join(str(x) for x in loc)}: {msg}")
            else:
                parts.append(str(msg))
            continue
        if item:
            parts.append(str(item))
    return parts


def _format_api_error(exc: BaseException) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        text = exc.response.text
        try:
            payload = exc.response.json()
            if isinstance(payload, dict) and "detail" in payload:
                detail = payload["detail"]
                if isinstance(detail, list):
                    parts = _format_validation_detail(detail)
                    text = "; ".join(parts) if parts else json.dumps(detail)
                else:
                    text = str(detail)
        except (ValueError, json.JSONDecodeError, TypeError):
            pass
        return f"HTTP {exc.response.status_code}: {text}"
    return str(exc)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
