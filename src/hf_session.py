"""Session-only Hugging Face auth. Never print or persist the token."""

from __future__ import annotations

import os
from typing import Any


def _looks_like_colab() -> bool:
    return bool(os.environ.get("COLAB_RELEASE_TAG")) or os.path.isdir("/content")


def session_hf_token() -> str | None:
    """Read token from env or Colab secret. Do not log the value."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(key)
        if val:
            return val
    if not _looks_like_colab():
        return None
    try:
        from google.colab import userdata  # type: ignore
    except ImportError:
        return None
    try:
        val = userdata.get("HF_TOKEN")
    except Exception:
        return None
    if val:
        os.environ["HF_TOKEN"] = val
    return val or None


def huggingface_from_pretrained_kwargs() -> dict[str, Any]:
    token = session_hf_token()
    if token:
        return {"token": token}
    return {}


def token_status() -> str:
    """Safe for logs: never includes the secret."""
    return "present" if session_hf_token() else "absent"
