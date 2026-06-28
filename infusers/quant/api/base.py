"""PyTorch Quant bridge — generator quants override forward_gen only."""

from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass
from typing import final

import torch.nn as nn
from reqm import Quant
from reqm.overrides_ext import allow_any_override, override


@dataclass(frozen=True)
class IntermediateEvent:
    message: str


@dataclass(frozen=True)
class FinalEvent:
    message: str


class TorchQuant[TIntermediate, TFinal](nn.Module, Quant):
    @override
    def __call__(self, **kwargs: object) -> object:
        return nn.Module.__call__(self, **kwargs)

    @override
    @abc.abstractmethod
    @allow_any_override
    def forward_gen(self, **kwargs: object) -> Iterator[TIntermediate | TFinal]: ...

    @final
    @override
    def forward(self, **kwargs: object) -> TFinal:
        final: TFinal | None = None
        for item in self.forward_gen(**kwargs):
            if isinstance(item, IntermediateEvent):
                continue
            final = item  # type: ignore[assignment]
        if final is None:
            raise RuntimeError(f"{type(self).__name__}.forward_gen did not yield a final result")
        return final
