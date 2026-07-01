"""Tests for atomic translators."""

from __future__ import annotations

import base64
import io

import pytest
import torch
from PIL import Image

from infusers.modal_app.translators.context import TranslatorContext
from infusers.modal_app.translators.dsl import apply


def _solid_red_png_b64() -> str:
    pil = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_imageb64_to_tensor_shape_and_range() -> None:
    ctx = TranslatorContext(device=torch.device("cpu"))
    tensor = apply("imageb64_to_tensor", _solid_red_png_b64(), ctx)
    assert tensor.shape == (3, 8, 8)
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0
    assert tensor[0].mean() > 0.9  # red channel dominant


def test_b64_roundtrip_red_pixel() -> None:
    ctx = TranslatorContext(device=torch.device("cpu"))
    b64_in = _solid_red_png_b64()
    tensor = apply("imageb64_to_tensor", b64_in, ctx)
    b64_out = apply("tensor_to_webp_b64", tensor, ctx)

    raw = base64.b64decode(b64_out)
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    r, g, b = pil.getpixel((4, 4))
    assert r > 200
    assert g < 50
    assert b < 50


def test_list_apply_imageb64_to_tensor() -> None:
    ctx = TranslatorContext(device=torch.device("cpu"))
    tensors = apply(
        "list_apply[imageb64_to_tensor]",
        [_solid_red_png_b64(), _solid_red_png_b64()],
        ctx,
    )
    assert len(tensors) == 2
    assert all(t.shape == (3, 8, 8) for t in tensors)


def test_imageb64_requires_device() -> None:
    with pytest.raises(RuntimeError, match="device is required"):
        apply("imageb64_to_tensor", _solid_red_png_b64(), TranslatorContext())


def _solid_rgba_png_b64(alpha: int = 255) -> str:
    pil = Image.new("RGBA", (8, 8), color=(10, 20, 30, alpha))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_rgba_b64_to_tensor_shape() -> None:
    ctx = TranslatorContext(device=torch.device("cpu"))
    tensor = apply("rgba_b64_to_tensor", _solid_rgba_png_b64(128), ctx)
    assert tensor.shape == (4, 8, 8)
    assert tensor[0, 0, 0].item() == pytest.approx(10 / 255.0, abs=0.02)
    assert tensor[3, 0, 0].item() == pytest.approx(128 / 255.0, abs=0.02)
