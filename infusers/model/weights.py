"""Resolve Klein checkpoint paths for Modal Volume vs local staging."""

from __future__ import annotations

from pathlib import Path

from flux2.util import FLUX2_MODEL_INFO

FLOW_FILENAME = "flux-2-klein-9b.safetensors"
FLOW_FILENAME_KV = "flux-2-klein-9b-kv.safetensors"
_AE_FILENAME = "ae.safetensors"
_MODAL_CKPT_ROOT = Path("/weights/klein-9b/klein-9b")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def local_ckpt_dir() -> Path:
    return repo_root() / "weights" / "klein-9b" / "klein-9b"


def flow_filename_for_model(model_name: str) -> str:
    """Return the flow safetensors filename for a flux2 Klein model name."""
    return FLUX2_MODEL_INFO[model_name.lower()]["filename"]


def resolve_weights_dir(
    weights_dir: Path | str | None = None,
    *,
    flow_filename: str = FLOW_FILENAME,
) -> Path:
    """Return the directory containing Klein flow + AE safetensors."""
    if weights_dir is not None:
        root = Path(weights_dir)
        if not (root / flow_filename).is_file():
            raise FileNotFoundError(
                f"Missing {flow_filename} under {root}. "
                "Run ./scripts/stage_weights.sh or ./scripts/upload_weights.sh."
            )
        return root

    modal_flow = _MODAL_CKPT_ROOT / flow_filename
    if modal_flow.is_file():
        return _MODAL_CKPT_ROOT

    local = local_ckpt_dir()
    if (local / flow_filename).is_file():
        return local

    raise FileNotFoundError(
        f"Klein weights not found for {flow_filename} at {_MODAL_CKPT_ROOT} (Modal) "
        f"or {local} (local).\n"
        "Local: ./scripts/stage_weights.sh\n"
        "Modal: ./scripts/upload_weights.sh"
    )


def resolve_hf_home(hf_home: Path | str | None = None) -> Path | None:
    """Return HF cache root for Qwen text encoder, if configured."""
    if hf_home is not None:
        root = Path(hf_home)
        if not root.is_dir():
            raise FileNotFoundError(f"HF cache directory not found: {root}")
        return root
    return None
