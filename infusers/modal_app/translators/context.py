"""Runtime context passed to translators during inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TranslatorContext:
    device: torch.device | None = None
