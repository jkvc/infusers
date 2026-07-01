"""Unit tests for signal_rgba VN SDEdit helpers."""

from __future__ import annotations

import pytest
import torch

from infusers.quant.flux.vnsdedit import (
    blend_with_signal_and_mask,
    compose_signal_rgba,
    compute_radial_mask,
    encode_signal_blend,
    split_signal_rgba,
    validate_signal_rgba,
)


class _FakeAE(torch.nn.Module):
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_validate_signal_rgba_rejects_wrong_size() -> None:
    signal = torch.zeros(4, 8, 8)
    with pytest.raises(ValueError, match="must match resolution"):
        validate_signal_rgba(signal, 16, 16)


def test_split_signal_rgba_channels() -> None:
    signal = torch.zeros(4, 2, 2)
    signal[0, 0, 0] = 1.0
    signal[1, 1, 1] = 1.0
    signal[2, :, :] = 0.5
    signal[3, 0, 1] = 0.8
    rgb, mask = split_signal_rgba(signal)
    assert rgb.shape == (3, 2, 2)
    assert mask.shape == (1, 2, 2)
    assert mask[0, 0, 1].item() == pytest.approx(0.8)


def test_compose_signal_rgba_roundtrip() -> None:
    rgb = torch.rand(3, 4, 4)
    mask = torch.rand(1, 4, 4)
    merged = compose_signal_rgba(rgb, mask)
    rgb2, mask2 = split_signal_rgba(merged)
    assert torch.allclose(rgb, rgb2)
    assert torch.allclose(mask, mask2)


def test_blend_preserves_low_mask_region() -> None:
    latent_h, latent_w = 2, 2
    channels = 2
    seq = latent_h * latent_w
    img = torch.ones(1, seq, channels)
    signal_latent = torch.zeros(1, channels, latent_h, latent_w)
    signal_mask_latent = torch.zeros(1, 1, latent_h, latent_w)
    noise = torch.full((1, channels, latent_h, latent_w), 2.0)
    out = blend_with_signal_and_mask(
        img,
        signal_latent,
        signal_mask_latent,
        noise,
        t_prev=0.5,
        latent_h=latent_h,
        latent_w=latent_w,
    )
    img_spatial = out.view(1, latent_h, latent_w, channels).permute(0, 3, 1, 2)
    assert torch.allclose(img_spatial, torch.full_like(img_spatial, 1.0))


def test_blend_uses_noised_signal_when_mask_high_and_t_prev_high() -> None:
    latent_h, latent_w = 1, 1
    channels = 1
    img = torch.zeros(1, 1, channels)
    signal_latent = torch.zeros(1, channels, latent_h, latent_w)
    signal_mask_latent = torch.ones(1, 1, latent_h, latent_w)
    noise = torch.ones(1, channels, latent_h, latent_w)
    out = blend_with_signal_and_mask(
        img,
        signal_latent,
        signal_mask_latent,
        noise,
        t_prev=1.0,
        latent_h=latent_h,
        latent_w=latent_w,
    )
    assert out.item() == pytest.approx(1.0)


def test_encode_signal_blend_resizes_mask() -> None:
    rgb = torch.rand(3, 64, 64)
    mask = torch.ones(1, 64, 64)
    noise = torch.zeros(1, 4, 2, 2)
    state = encode_signal_blend(
        _FakeAE(),
        rgb,
        mask,
        noise,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert state.signal_latent.shape == (1, 3, 64, 64)
    assert state.signal_mask_latent.shape == (1, 1, 2, 2)


def test_radial_mask_peak_at_center() -> None:
    mask = compute_radial_mask(
        width=64,
        height=64,
        click_x=32,
        click_y=32,
        edit_max=1.0,
        radius_px=20.0,
        core_radius_px=5.0,
        falloff_shape="cosine",
    )
    assert mask.shape == (1, 64, 64)
    assert mask[0, 32, 32].item() == pytest.approx(1.0)
    assert mask[0, 0, 0].item() == pytest.approx(0.0, abs=1e-6)


def test_radial_mask_gaussian_falloff() -> None:
    mask = compute_radial_mask(
        width=32,
        height=32,
        click_x=16,
        click_y=16,
        edit_max=1.0,
        radius_px=16.0,
        core_radius_px=0.0,
        falloff_shape="gaussian",
    )
    mid = mask[0, 16, 20].item()
    assert 0.0 < mid < 1.0
    assert mask[0, 16, 16].item() == pytest.approx(1.0)
