"""Optional API key enforcement for HTTP requests."""

from __future__ import annotations

import hmac

from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import Settings


def path_exempt_from_api_key(path: str, settings: Settings) -> bool:
    return path in ("/", "/health", "/v1/health") or (
        settings.api_key_exempt_docs
        and (path.startswith("/docs") or path.startswith("/redoc") or path == "/openapi.json")
    )


def api_key_rejection(request: Request, settings: Settings) -> JSONResponse | None:
    key = (settings.api_key or "").strip()
    if not key:
        return None
    if request.method == "OPTIONS":
        return None
    if path_exempt_from_api_key(request.url.path, settings):
        return None
    auth = request.headers.get("authorization") or ""
    bearer = ""
    if auth.lower().startswith("bearer "):
        bearer = auth[7:].strip()
    header_key = request.headers.get("x-api-key") or ""
    if hmac.compare_digest(bearer, key) or hmac.compare_digest(header_key, key):
        return None
    return JSONResponse(
        status_code=401,
        content={"detail": "Invalid or missing API key"},
    )
