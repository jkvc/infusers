"""Klein 9B model — weights and modules only (no inference hyperparams)."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn
from flux2.text_encoder import Qwen3Embedder
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder

from infusers.model.weights import (
    flow_filename_for_model,
    resolve_hf_home,
    resolve_weights_dir,
)

_AE_FILENAME = "ae.safetensors"
_FP8_MIN_CAPABILITY = (8, 9)


def _qwen_variant_for_model(model_name: str) -> str:
    name = model_name.lower()
    if "4b" in name:
        return "4B"
    if "9b" in name:
        return "8B"
    raise ValueError(f"No Qwen3 fallback mapping for model {model_name!r}")


def _configure_preseeded_weights(weights_dir: Path, model_name: str) -> None:
    model_info = FLUX2_MODEL_INFO[model_name.lower()]
    flow_filename = model_info["filename"]
    model_path_env = model_info["model_path"]
    flow_path = weights_dir / flow_filename
    ae_path = weights_dir / _AE_FILENAME
    missing = [path for path in (flow_path, ae_path) if not path.is_file()]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Pre-seeded Klein weights not found:\n{missing_list}")
    os.environ[model_path_env] = str(flow_path)
    os.environ["AE_MODEL_PATH"] = str(ae_path)


def _load_text_encoder_compat(model_name: str, device: torch.device) -> Qwen3Embedder:
    cap = torch.cuda.get_device_capability()
    if cap >= _FP8_MIN_CAPABILITY:
        return load_text_encoder(model_name, device=device)

    if "klein" not in model_name.lower():
        return load_text_encoder(model_name, device=device)

    variant = _qwen_variant_for_model(model_name)
    model_spec = f"Qwen/Qwen3-{variant}"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    embedder = Qwen3Embedder.__new__(Qwen3Embedder)
    nn.Module.__init__(embedder)
    embedder.model = AutoModelForCausalLM.from_pretrained(
        model_spec,
        torch_dtype=torch.bfloat16,
        device_map=str(device),
    )
    embedder.tokenizer = AutoTokenizer.from_pretrained(model_spec)
    embedder.max_length = 512
    return embedder.eval()


def _load_klein_modules(
    *,
    model_name: str,
    device: torch.device,
    flow_device: torch.device,
    parallel_load: bool,
) -> tuple[Qwen3Embedder, nn.Module, nn.Module]:
    """Load text encoder, flow, and VAE — optionally in parallel (I/O overlap)."""

    def _text_encoder() -> Qwen3Embedder:
        return _load_text_encoder_compat(model_name, device=device)

    def _flow() -> nn.Module:
        return load_flow_model(model_name, device=flow_device)

    def _ae() -> nn.Module:
        return load_ae(model_name, device=device)

    if not parallel_load:
        return _text_encoder(), _flow(), _ae()

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="klein-load") as pool:
        fut_te = pool.submit(_text_encoder)
        fut_flow = pool.submit(_flow)
        fut_ae = pool.submit(_ae)
        return fut_te.result(), fut_flow.result(), fut_ae.result()


class KleinModel(nn.Module):
    """Loaded Klein flow, VAE, and text encoder — no steps/guidance/resolution state."""

    model_name: str
    model_info: dict
    device: torch.device

    def __init__(
        self,
        model_name: str = "flux.2-klein-9b",
        weights_dir: str | Path | None = None,
        hf_home: str | Path | None = None,
        load_flow_on_cpu: bool = False,
        parallel_load: bool = True,
    ) -> None:
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for KleinModel")

        ckpt_dir = resolve_weights_dir(
            weights_dir,
            flow_filename=flow_filename_for_model(model_name),
        )
        _configure_preseeded_weights(ckpt_dir, model_name.lower())

        hf_root = resolve_hf_home(hf_home)
        if hf_root is not None:
            os.environ["HF_HOME"] = str(hf_root)

        self.model_name = model_name.lower()
        self.model_info = FLUX2_MODEL_INFO[self.model_name]
        self.device = torch.device("cuda")

        flow_device = torch.device("cpu") if load_flow_on_cpu else self.device
        text_encoder, flow, ae = _load_klein_modules(
            model_name=self.model_name,
            device=self.device,
            flow_device=flow_device,
            parallel_load=parallel_load,
        )
        text_encoder.eval()
        flow.eval()
        ae.eval()

        self.text_encoder = text_encoder
        self.flow = flow
        self.ae = ae
