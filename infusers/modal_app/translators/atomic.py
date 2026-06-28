"""Atomic wire-format translators."""

from __future__ import annotations

import base64
import io
from typing import Any

import torch
from PIL import Image

from infusers.modal_app.translators.context import TranslatorContext
from infusers.modal_app.translators.registry import register
from infusers.quant.api.image_base import chw_float01_to_pil, pil_to_chw_float01


def _require_device(ctx: TranslatorContext) -> torch.device:
    if ctx.device is None:
        raise RuntimeError("TranslatorContext.device is required for tensor conversion")
    return ctx.device


class Identity:
    def __call__(self, value: Any, _ctx: TranslatorContext) -> Any:
        return value


class GetAttr:
    def __init__(self, field: str) -> None:
        self.field = field

    def __call__(self, value: Any, _ctx: TranslatorContext) -> Any:
        if isinstance(value, dict) and self.field in value:
            return value[self.field]
        if hasattr(value, self.field):
            return getattr(value, self.field)
        raise TypeError(f"Cannot get attribute {self.field!r} from {type(value).__name__}")

    def __repr__(self) -> str:
        return f"GetAttr({self.field!r})"


class ImageB64ToTensor:
    def __call__(self, value: str, ctx: TranslatorContext) -> torch.Tensor:
        device = _require_device(ctx)
        raw = base64.b64decode(value)
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        return pil_to_chw_float01(pil, device)


class TensorToWebpB64:
    def __call__(self, value: torch.Tensor, _ctx: TranslatorContext) -> str:
        pil = chw_float01_to_pil(value)
        buf = io.BytesIO()
        pil.save(buf, format="WEBP", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")


@register("identity")
def _identity_dsl(_value: str | None = None) -> Identity:
    return Identity()


@register("get_attr")
def _get_attr_dsl(field: str) -> GetAttr:
    return GetAttr(field)


@register("imageb64_to_tensor")
def _imageb64_to_tensor_dsl(_value: str | None = None) -> ImageB64ToTensor:
    return ImageB64ToTensor()


@register("tensor_to_webp_b64")
def _tensor_to_webp_b64_dsl(_value: str | None = None) -> TensorToWebpB64:
    return TensorToWebpB64()
