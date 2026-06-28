"""PyTorch Quant bridge — copy of reqm torch_quant recipe."""

from __future__ import annotations

import abc

import torch.nn as nn
from reqm import Quant
from reqm.overrides_ext import allow_any_override, override


class TorchQuant(nn.Module, Quant):
    @override
    def __call__(self, **kwargs: object) -> object:
        return nn.Module.__call__(self, **kwargs)

    @override
    @abc.abstractmethod
    @allow_any_override
    def forward(self, **kwargs: object) -> object: ...
