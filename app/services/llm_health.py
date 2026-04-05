"""Lightweight LLM reachability checks for /health (optional)."""

from __future__ import annotations

import httpx

from app.config import LLMProvider, Settings

_LLM_HEALTH_TIMEOUT_S = 5.0


def probe_llm(settings: Settings) -> tuple[bool, str | None]:
    """
    Return (ok, error_code) where error_code is timeout|unavailable|unknown or None.
    Does not raise; avoids leaking secrets in logs from this function.
    """
    timeout = min(_LLM_HEALTH_TIMEOUT_S, max(1.0, settings.openai_timeout_seconds))
    if settings.llm_provider == LLMProvider.OLLAMA:
        return _probe_ollama(settings, timeout)
    return _probe_openai(settings, timeout)


def _probe_ollama(settings: Settings, timeout: float) -> tuple[bool, str | None]:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(url)
        if r.is_success:
            return True, None
        return False, "unavailable"
    except httpx.TimeoutException:
        return False, "timeout"
    except httpx.RequestError:
        return False, "unavailable"
    except Exception:  # pylint: disable=broad-exception-caught
        return False, "unknown"


def _probe_openai(settings: Settings, timeout: float) -> tuple[bool, str | None]:
    if not settings.openai_api_key:
        return False, "unavailable"
    url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(url, headers=headers)
        if r.is_success:
            return True, None
        return False, "unavailable"
    except httpx.TimeoutException:
        return False, "timeout"
    except httpx.RequestError:
        return False, "unavailable"
    except Exception:  # pylint: disable=broad-exception-caught
        return False, "unknown"
