"""GPU smoke test for FluxPanoramaQuant — opt-in via pytest -m gpu."""

from __future__ import annotations

import pytest
import torch

from infusers import QM
from infusers.quant.api.pano_base import PanoramaOutput
from infusers.quant.flux.panorama import FluxPanoramaQuant, compute_canvas_dims

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def skip_without_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def test_pano_recipe_builds(skip_without_gpu: None) -> None:
    quant = QM.build("quant/flux/klein9b/pano_basic")
    assert isinstance(quant, FluxPanoramaQuant)


def test_pano_two_slice_horizontal_smoke(skip_without_gpu: None) -> None:
    quant = QM.build("quant/flux/klein9b/pano_basic")
    out = quant(
        prompts=["warm desert sand dunes", "cool blue ocean horizon"],
        seed=0,
        resolution=[256, 512],
        pano_direction="horizontal",
        overlap_pixels=128,
        num_steps=2,
    )
    assert isinstance(out, PanoramaOutput)
    dims = compute_canvas_dims(2, 256, 512, 128, "horizontal")
    assert out.output_size == [dims.output_height, dims.output_width]
    assert out.images.shape == (1, 3, dims.output_height, dims.output_width)
