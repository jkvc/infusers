"""Latent-space signal pasteback blending for image quants."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from flux2.sampling import default_prep
from torch import Tensor

from infusers.quant.api.image_base import chw_float01_to_pil


@dataclass(frozen=True)
class SignalBlendState:
    signal_latent: Tensor
    signal_mask_latent: Tensor
    noise_spatial: Tensor
    latent_h: int
    latent_w: int


def validate_signal_rgba(signal_rgba: Tensor, height: int, width: int) -> None:
    if signal_rgba.ndim != 3:
        raise ValueError(f"signal_rgba must be CHW, got shape {tuple(signal_rgba.shape)}")
    if signal_rgba.shape[0] != 4:
        raise ValueError(f"signal_rgba must have 4 channels, got {signal_rgba.shape[0]}")
    if signal_rgba.shape[1] != height or signal_rgba.shape[2] != width:
        raise ValueError(
            f"signal_rgba spatial size {list(signal_rgba.shape[1:])} must match "
            f"resolution [{height}, {width}]"
        )


def split_signal_rgba(signal_rgba: Tensor) -> tuple[Tensor, Tensor]:
    validate_signal_rgba(signal_rgba, signal_rgba.shape[1], signal_rgba.shape[2])
    rgb = signal_rgba[:3].clamp(0.0, 1.0)
    mask = signal_rgba[3:4].clamp(0.0, 1.0)
    return rgb, mask


def blend_with_signal_and_mask(
    img: Tensor,
    signal_latent: Tensor,
    signal_mask_latent: Tensor,
    noise: Tensor,
    t_prev: float,
    latent_h: int,
    latent_w: int,
) -> Tensor:
    """Blend packed latents after an Euler step.

    Mask semantics: higher values → edit freely; lower values → preserve signal.
    """
    img_spatial = rearrange(img, "b (h w) c -> b c h w", h=latent_h, w=latent_w)
    noised_signal = signal_latent * (1.0 - t_prev) + noise * t_prev
    noised_signal = noised_signal.to(img_spatial.dtype)
    pasteback_mask = (signal_mask_latent <= t_prev).to(img_spatial.dtype)
    img_spatial = img_spatial * (1.0 - pasteback_mask) + noised_signal * pasteback_mask
    return rearrange(img_spatial, "b c h w -> b (h w) c")


def encode_signal_blend(
    ae: torch.nn.Module,
    signal_rgb: Tensor,
    signal_mask: Tensor,
    noise_spatial: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> SignalBlendState:
    latent_h = noise_spatial.shape[-2]
    latent_w = noise_spatial.shape[-1]
    prepped = default_prep(chw_float01_to_pil(signal_rgb.detach().cpu()), limit_pixels=None)
    if not isinstance(prepped, torch.Tensor):
        raise TypeError("default_prep must return a tensor for a single signal image")
    signal_nchw = prepped.unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.autocast(device_type=device.type, enabled=False):
        signal_latent = ae.encode(signal_nchw).to(dtype=dtype)
    mask_nchw = signal_mask.unsqueeze(0).to(device=device, dtype=torch.float32)
    signal_mask_latent = F.interpolate(
        mask_nchw,
        size=(latent_h, latent_w),
        mode="bilinear",
        align_corners=False,
    )
    return SignalBlendState(
        signal_latent=signal_latent,
        signal_mask_latent=signal_mask_latent,
        noise_spatial=noise_spatial,
        latent_h=latent_h,
        latent_w=latent_w,
    )


def compute_radial_mask(
    width: int,
    height: int,
    click_x: int,
    click_y: int,
    edit_max: float,
    radius_px: float,
    core_radius_px: float,
    falloff_shape: str = "cosine",
) -> Tensor:
    """Radial edit mask in [0, 1]. Matches localized-variation falloff shapes."""
    edit_max_clamped = max(0.0, min(1.0, edit_max))
    fully_zero = radius_px <= 0 or edit_max_clamped <= 0
    hard_disk = not fully_zero and core_radius_px >= radius_px
    effective_core = max(0.0, core_radius_px)
    denom = radius_px - effective_core

    mask = torch.zeros(height, width, dtype=torch.float32)
    for y in range(height):
        dy = y - click_y
        for x in range(width):
            dx = x - click_x
            dist = math.sqrt(dx * dx + dy * dy)
            if fully_zero:
                value = 0.0
            elif dist <= effective_core:
                value = edit_max_clamped
            elif dist >= radius_px:
                value = 0.0
            elif hard_disk:
                value = edit_max_clamped
            else:
                t = (dist - effective_core) / denom
                if falloff_shape == "linear":
                    weight = 1.0 - t
                elif falloff_shape == "gaussian":
                    weight = math.exp(-((t * 3.0) ** 2))
                else:
                    weight = 0.5 * (1.0 + math.cos(math.pi * t))
                value = edit_max_clamped * weight
            mask[y, x] = value
    return mask.unsqueeze(0)


def compose_signal_rgba(signal_rgb: Tensor, signal_mask: Tensor) -> Tensor:
    if signal_rgb.shape[1:] != signal_mask.shape[1:]:
        raise ValueError(
            f"signal_rgb spatial {tuple(signal_rgb.shape[1:])} must match mask "
            f"{tuple(signal_mask.shape[1:])}"
        )
    if signal_rgb.shape[0] != 3:
        raise ValueError(f"signal_rgb must have 3 channels, got {signal_rgb.shape[0]}")
    if signal_mask.shape[0] != 1:
        raise ValueError(f"signal_mask must have 1 channel, got {signal_mask.shape[0]}")
    return torch.cat([signal_rgb, signal_mask], dim=0)
