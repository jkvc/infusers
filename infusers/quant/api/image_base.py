"""Image quant domain contract and tensor helpers."""

from __future__ import annotations

import abc
import random
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from PIL import Image
from reqm.overrides_ext import override

from infusers.quant.api.base import FinalEvent, IntermediateEvent, TorchQuant


@dataclass(frozen=True)
class ImageIntermediateEvent(IntermediateEvent):
    pass


@dataclass(frozen=True)
class ImageOutput(FinalEvent):
    image: torch.Tensor  # float32 CHW [0, 1] on quant device


def pil_to_chw_float01(image: Image.Image, device: torch.device) -> torch.Tensor:
    import numpy as np

    arr = torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32) / 255.0)
    return arr.permute(2, 0, 1).to(device)


def pil_rgba_to_chw_float01(image: Image.Image, device: torch.device) -> torch.Tensor:
    import numpy as np

    arr = torch.from_numpy(np.array(image.convert("RGBA"), dtype=np.float32) / 255.0)
    return arr.permute(2, 0, 1).to(device)


def chw_float01_to_pil(tensor: torch.Tensor) -> Image.Image:
    import numpy as np

    t = tensor.detach().float().cpu().clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


class ImageQuant(TorchQuant[ImageIntermediateEvent, ImageOutput]):
    """Image quant — override forward_gen only; forward() drains the stream."""

    @override
    @abc.abstractmethod
    def forward_gen(
        self,
        prompt: str,
        seed: int | None = None,
        resolution: list[int] | None = None,
        cond_images: list[torch.Tensor] | None = None,
        signal_rgba: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> Iterator[ImageIntermediateEvent | ImageOutput]: ...

    @override
    @abc.abstractmethod
    def dummy_inputs(self) -> list[dict[str, object]]: ...


class DummyImageQuant(ImageQuant):
    """Trivial CPU-only image quant for tests — solid fill from seed, no model weights."""

    def __init__(self, num_steps: int = 3, resolution: list[int] | None = None) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.resolution = resolution or [64, 64]

    @override
    def forward_gen(
        self,
        prompt: str,
        seed: int | None = None,
        resolution: list[int] | None = None,
        cond_images: list[torch.Tensor] | None = None,
        signal_rgba: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> Iterator[ImageIntermediateEvent | ImageOutput]:
        height, width = resolution or self.resolution
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        yield ImageIntermediateEvent(message="dummy: begin")
        for step in range(1, self.num_steps + 1):
            yield ImageIntermediateEvent(message=f"dummy: step {step}/{self.num_steps}")

        rng = random.Random(seed)
        color = [rng.random() for _ in range(3)]
        chw = torch.tensor(color, dtype=torch.float32).view(3, 1, 1).expand(3, height, width)
        yield ImageOutput(message="dummy: done", image=chw)

    @override
    def dummy_inputs(self) -> list[dict[str, object]]:
        return [
            {
                "prompt": "dummy gray",
                "seed": 0,
                "resolution": self.resolution,
                "cond_images": None,
            }
        ]
