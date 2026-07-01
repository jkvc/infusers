"""Unit tests for panorama dimension math and helper tensors."""

from __future__ import annotations

import pytest
import torch

from infusers.quant.flux.panorama import (
    build_lerp_mask_slice,
    compute_canvas_dims,
    fold_wraparound_horizontal,
    fold_wraparound_vertical,
    vae_decode_crop_slices,
    validate_inputs,
)


def test_compute_canvas_dims_horizontal_three_slices() -> None:
    dims = compute_canvas_dims(
        num_slices=3,
        slice_height=512,
        slice_width=1024,
        overlap_pixels=256,
        pano_direction="horizontal",
    )
    assert dims.output_height == 512
    assert dims.output_width == (1024 - 256) * 3
    assert dims.overlap_latent == 16
    assert dims.latent_full_height == 32
    assert dims.latent_full_width == 144


def test_compute_canvas_dims_vertical_two_slices() -> None:
    dims = compute_canvas_dims(
        num_slices=2,
        slice_height=512,
        slice_width=1024,
        overlap_pixels=256,
        pano_direction="vertical",
    )
    assert dims.output_width == 1024
    assert dims.output_height == (512 - 256) * 2
    assert dims.latent_full_width == 64
    assert dims.latent_full_height == 16 * 2


def test_compute_canvas_dims_single_slice_degenerate() -> None:
    dims = compute_canvas_dims(
        num_slices=1,
        slice_height=512,
        slice_width=1024,
        overlap_pixels=256,
        pano_direction="horizontal",
    )
    assert dims.output_width == 1024 - 256


def test_validate_overlap_not_divisible_by_16() -> None:
    with pytest.raises(ValueError, match="divisible by 16"):
        validate_inputs(["a"], [512, 1024], 130, "horizontal", None)


def test_validate_overlap_zero() -> None:
    with pytest.raises(ValueError, match="overlap_pixels must be > 0"):
        validate_inputs(["a"], [512, 1024], 0, "horizontal", None)


def test_validate_overlap_too_large() -> None:
    with pytest.raises(ValueError, match="must be <"):
        validate_inputs(["a"], [512, 1024], 1024, "horizontal", None)


def test_validate_resolution_not_divisible_by_16() -> None:
    with pytest.raises(ValueError, match="divisible by 16"):
        validate_inputs(["a"], [513, 1024], 256, "horizontal", None)


def test_validate_per_slice_cond_length_mismatch() -> None:
    with pytest.raises(ValueError, match="outer list length"):
        validate_inputs(
            ["a", "b"],
            [512, 1024],
            256,
            "horizontal",
            [[torch.zeros(3, 64, 64)]],
        )


def test_lerp_mask_horizontal_shape_and_ramps() -> None:
    mask = build_lerp_mask_slice(
        slice_height_latent=8,
        slice_width_latent=16,
        overlap_latent=4,
        pano_direction="horizontal",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mask.shape == (1, 1, 8, 16)
    assert mask[0, 0, 0, 0].item() == pytest.approx(0.0)
    assert mask[0, 0, 0, 3].item() == pytest.approx(1.0)
    assert mask[0, 0, 0, 8].item() == pytest.approx(1.0)
    assert mask[0, 0, 0, -1].item() == pytest.approx(0.0)


def test_lerp_mask_vertical_shape() -> None:
    mask = build_lerp_mask_slice(
        slice_height_latent=16,
        slice_width_latent=8,
        overlap_latent=4,
        pano_direction="vertical",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mask.shape == (1, 1, 16, 8)


def test_fold_wraparound_horizontal() -> None:
    overlap = 2
    full_w = 6
    wrap_w = full_w + overlap
    pred = torch.zeros(1, 1, 1, wrap_w)
    pred[0, 0, 0, -overlap:] = torch.tensor([1.0, 2.0])
    pred[0, 0, 0, :full_w] = 10.0
    folded = fold_wraparound_horizontal(pred, full_w, overlap)
    assert folded.shape[-1] == full_w
    assert folded[0, 0, 0, 0] == 11.0
    assert folded[0, 0, 0, 1] == 12.0


def test_fold_wraparound_vertical() -> None:
    overlap = 2
    full_h = 5
    wrap_h = full_h + overlap
    pred = torch.zeros(1, 1, wrap_h, 1)
    pred[0, 0, -overlap:, 0] = torch.tensor([3.0, 4.0])
    pred[0, 0, :full_h, 0] = 7.0
    folded = fold_wraparound_vertical(pred, full_h, overlap)
    assert folded[0, 0, 0, 0] == 10.0
    assert folded[0, 0, 1, 0] == 11.0


def test_vae_decode_crop_slices_horizontal() -> None:
    h_slice, w_slice = vae_decode_crop_slices(
        pano_direction="horizontal",
        pad_latent=16,
        pad_apparent=256,
        true_apparent_width=2304,
        true_apparent_height=512,
    )
    assert h_slice == slice(None)
    assert w_slice == slice(256, 256 + 2304)


def test_vae_decode_crop_slices_vertical() -> None:
    h_slice, w_slice = vae_decode_crop_slices(
        pano_direction="vertical",
        pad_latent=16,
        pad_apparent=256,
        true_apparent_width=1024,
        true_apparent_height=512,
    )
    assert h_slice == slice(256, 256 + 512)
    assert w_slice == slice(None)
