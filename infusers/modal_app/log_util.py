"""Structured print logging for Modal runner requests."""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any

_LOG_PREFIX = "[runner]"


def log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    print(f"{_LOG_PREFIX} {json.dumps(payload, indent=2, default=_json_default)}", flush=True)


def summarize_request(body: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"path": body.get("path")}
    inputs = body.get("inputs")
    if isinstance(inputs, dict):
        out["inputs"] = _summarize_value(inputs)
    translator = body.get("translator")
    if translator:
        out["translator"] = translator
    return out


def summarize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return _summarize_value(kwargs)


def summarize_output_value(key: str, value: Any) -> dict[str, Any]:
    summary = _summarize_value(value)
    return {"key": key, "value": summary}


def _summarize_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        return {k: _summarize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summarize_value(v) for v in value]
    if isinstance(value, str):
        return _summarize_string(value)
    return _json_default(value)


def _summarize_string(value: str) -> str | dict[str, Any]:
    if len(value) <= 120 and not _looks_like_base64(value):
        return value
    decoded_len = _decoded_byte_length(value)
    if decoded_len is not None:
        return {
            "kind": "base64",
            "chars": len(value),
            "bytes": decoded_len,
        }
    return {"kind": "string", "chars": len(value)}


def _looks_like_base64(value: str) -> bool:
    if len(value) < 32:
        return False
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n")
    return all(ch in allowed for ch in value[:256])


def _decoded_byte_length(value: str) -> int | None:
    if not _looks_like_base64(value):
        return None
    try:
        return len(base64.b64decode(value, validate=True))
    except (binascii.Error, ValueError):
        try:
            return len(base64.b64decode(value))
        except (binascii.Error, ValueError):
            return None


def _json_default(value: Any) -> Any:
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    if torch is not None and isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if hasattr(value, "__dataclass_fields__"):
        return {k: _summarize_value(getattr(value, k)) for k in value.__dataclass_fields__}
    return {"kind": type(value).__name__, "repr": repr(value)[:200]}
