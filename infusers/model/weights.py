"""Resolve Klein checkpoint paths for Modal Volume vs local staging."""

from __future__ import annotations

from pathlib import Path

FLOW_FILENAME = "flux-2-klein-9b.safetensors"
MODAL_CKPT = Path("/weights/klein-9b/klein-9b") / FLOW_FILENAME


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def local_ckpt_dir() -> Path:
    return repo_root() / "weights" / "klein-9b" / "klein-9b"


def resolve_weights_dir(weights_dir: Path | str | None = None) -> Path:
    """Return the directory containing Klein flow + AE safetensors."""
    if weights_dir is not None:
        root = Path(weights_dir)
        if not (root / FLOW_FILENAME).is_file():
            raise FileNotFoundError(
                f"Missing {FLOW_FILENAME} under {root}. "
                "Run ./scripts/stage_weights.sh or ./scripts/upload_weights.sh."
            )
        return root

    if MODAL_CKPT.is_file():
        return MODAL_CKPT.parent

    local = local_ckpt_dir()
    if (local / FLOW_FILENAME).is_file():
        return local

    raise FileNotFoundError(
        f"Klein 9B weights not found at {MODAL_CKPT} (Modal) or {local} (local).\n"
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
