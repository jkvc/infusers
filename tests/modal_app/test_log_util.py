"""Tests for runner log summarization."""

from __future__ import annotations

import base64
import io

from PIL import Image

from infusers.modal_app.log_util import summarize_request


def test_summarize_request_redacts_base64() -> None:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    summary = summarize_request(
        {
            "path": "klein9b.image",
            "inputs": {
                "prompt": "hello",
                "cond_images": [b64],
            },
            "translator": {"cond_images": "list_apply[imageb64_to_tensor]"},
        }
    )

    assert summary["path"] == "klein9b.image"
    assert summary["inputs"]["prompt"] == "hello"
    cond = summary["inputs"]["cond_images"][0]
    assert cond["kind"] == "base64"
    assert cond["bytes"] > 0
    assert cond["chars"] == len(b64)
    assert b64 not in str(summary)
