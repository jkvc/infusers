"""Panorama quant domain contract."""

from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from reqm.overrides_ext import override

from infusers.quant.api.base import FinalEvent, IntermediateEvent, TorchQuant

# Document-only shape hints — not enforced at runtime.
type CHWTensor = torch.Tensor
type NCHWTensor = torch.Tensor


@dataclass(frozen=True)
class PanoramaIntermediateEvent(IntermediateEvent):
    pass


@dataclass(frozen=True)
class PanoramaOutput(FinalEvent):
    images: NCHWTensor  # float32 (N, C, H, W) in [0, 1] on quant device; N=1 for single pano
    direction: str
    slice_resolution: list[int]
    output_size: list[int]
    num_slices: int
    overlap_pixels: int


class PanoramaQuant(TorchQuant[PanoramaIntermediateEvent, PanoramaOutput]):
    """Panorama quant — override forward_gen only; forward() drains the stream."""

    @override
    @abc.abstractmethod
    def forward_gen(
        self,
        prompts: list[str],
        seed: int | None = None,
        resolution: list[int] | None = None,
        pano_direction: str | None = None,
        overlap_pixels: int | None = None,
        cond_images: list[torch.Tensor] | list[list[torch.Tensor]] | None = None,
        num_steps: int | None = None,
    ) -> Iterator[PanoramaIntermediateEvent | PanoramaOutput]: ...

    @override
    @abc.abstractmethod
    def dummy_inputs(self) -> list[dict[str, object]]: ...
