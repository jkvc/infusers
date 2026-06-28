"""Image quant domain contract and tensor helpers."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch
from PIL import Image
from reqm.overrides_ext import override

from infusers.quant.base import TorchQuant


@dataclass
class ImageOutput:
    image: torch.Tensor  # float32 CHW [0, 1] on quant device


def pil_to_chw_float01(image: Image.Image, device: torch.device) -> torch.Tensor:
    import numpy as np

    arr = torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32) / 255.0)
    return arr.permute(2, 0, 1).to(device)


def chw_float01_to_pil(tensor: torch.Tensor) -> Image.Image:
    import numpy as np

    t = tensor.detach().float().cpu().clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


class ImageQuant(TorchQuant):
    """Abstract image inferencer — locked forward signature for uniform call sites."""

    @override
    @abc.abstractmethod
    def forward(
        self,
        prompt: str,
        seed: int | None = None,
        resolution: list[int] | None = None,
        cond_images: list[torch.Tensor] | None = None,
    ) -> ImageOutput: ...

    @override
    @abc.abstractmethod
    def dummy_inputs(self) -> list[dict[str, object]]: ...
