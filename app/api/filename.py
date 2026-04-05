"""Validate document names for path traversal safety."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException


def require_safe_filename(name: str) -> str:
    if not name or name != name.strip():
        raise HTTPException(status_code=400, detail="Invalid filename")
    if "\x00" in name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if any(x in name for x in ("/", "\\", "..")):
        raise HTTPException(status_code=400, detail="Invalid filename")
    base = Path(name).name
    if not base or base != name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return base
