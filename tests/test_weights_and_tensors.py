from pathlib import Path

import pytest
import torch
from PIL import Image

from infusers.model.weights import FLOW_FILENAME, local_ckpt_dir, resolve_weights_dir
from infusers.quant.api.image_base import chw_float01_to_pil, pil_to_chw_float01


def test_resolve_weights_dir_explicit(tmp_path: Path) -> None:
    (tmp_path / FLOW_FILENAME).write_bytes(b"x")
    (tmp_path / "ae.safetensors").write_bytes(b"x")
    assert resolve_weights_dir(tmp_path) == tmp_path


def test_resolve_weights_dir_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="stage_weights"):
        resolve_weights_dir(tmp_path)


def test_local_ckpt_dir_under_repo() -> None:
    root = local_ckpt_dir()
    assert root.name == "klein-9b"
    assert root.parent.name == "klein-9b"


def test_chw_pil_roundtrip() -> None:
    pil = Image.new("RGB", (8, 4), color=(128, 64, 32))
    t = pil_to_chw_float01(pil, torch.device("cpu"))
    assert t.shape == (3, 4, 8)
    assert t.min() >= 0.0 and t.max() <= 1.0
    back = chw_float01_to_pil(t)
    assert back.size == (8, 4)
