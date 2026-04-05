"""Unit tests for UI helper functions (no Streamlit dependency)."""

from __future__ import annotations

import httpx

from ui.streamlit_app import _format_api_error, _req_headers


def _fake_http_status_error(
    status_code: int, body: str
) -> httpx.HTTPStatusError:
    response = httpx.Response(status_code, text=body)
    request = httpx.Request("POST", "http://test/query")
    return httpx.HTTPStatusError(
        f"{status_code}", request=request, response=response
    )


def test_format_api_error_with_detail():
    exc = _fake_http_status_error(400, '{"detail":"boom"}')
    result = _format_api_error(exc)
    assert result.startswith("HTTP 400")
    assert "boom" in result


def test_format_api_error_plain_text_body():
    exc = _fake_http_status_error(502, "Bad Gateway")
    result = _format_api_error(exc)
    assert result.startswith("HTTP 502")
    assert "Bad Gateway" in result


def test_format_api_error_generic_exception():
    exc = ValueError("something broke")
    result = _format_api_error(exc)
    assert result == "something broke"


def test_req_headers_has_request_id():
    headers = _req_headers()
    assert "X-Request-ID" in headers
    assert len(headers["X-Request-ID"]) > 0


def test_req_headers_includes_api_key(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "abc123")
    headers = _req_headers()
    assert headers.get("X-API-Key") == "abc123"


def test_req_headers_no_api_key_when_unset(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    headers = _req_headers()
    assert "X-API-Key" not in headers
