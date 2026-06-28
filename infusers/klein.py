"""Klein inference via BFL flux2 — shared between Modal and local scripts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from einops import rearrange
from flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise,
    denoise_cached,
    denoise_cfg,
    encode_image_refs,
    get_schedule,
    scatter_ids,
)
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder
from PIL import Image

DEFAULT_WEIGHTS_DIR = Path("/weights/klein-9b/klein-9b")
FLOW_FILENAME = "flux-2-klein-9b.safetensors"
AE_FILENAME = "ae.safetensors"


def configure_preseeded_weights(weights_dir: Path | str = DEFAULT_WEIGHTS_DIR) -> Path:
    """Point BFL flux2 loaders at pre-uploaded safetensors (Modal Volume, etc.)."""
    root = Path(weights_dir)
    flow_path = root / FLOW_FILENAME
    ae_path = root / AE_FILENAME
    missing = [path for path in (flow_path, ae_path) if not path.is_file()]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Pre-seeded Klein 9B weights not found under {root}:\n{missing_list}"
        )
    os.environ["KLEIN_9B_MODEL_PATH"] = str(flow_path)
    os.environ["AE_MODEL_PATH"] = str(ae_path)
    return root


@dataclass
class KleinPipeline:
    model_name: str
    model_info: dict
    device: torch.device
    text_encoder: torch.nn.Module
    model: torch.nn.Module
    ae: torch.nn.Module
    num_steps: int
    guidance: float
    width: int
    height: int
    ref_tokens: torch.Tensor | None
    ref_ids: torch.Tensor | None


def load_pipeline(
    model_name: str = "flux.2-klein-9b",
    *,
    width: int = 1024,
    height: int = 1024,
    weights_dir: Path | str | None = DEFAULT_WEIGHTS_DIR,
    ref_images: list[Image.Image] | None = None,
) -> KleinPipeline:
    if weights_dir is not None:
        configure_preseeded_weights(weights_dir)

    model_name = model_name.lower()
    model_info = FLUX2_MODEL_INFO[model_name]
    defaults = model_info.get("defaults", {})
    device = torch.device("cuda")

    text_encoder = load_text_encoder(model_name, device=device)
    model = load_flow_model(model_name, device=device)
    ae = load_ae(model_name, device=device)
    text_encoder.eval()
    model.eval()
    ae.eval()

    ref_tokens = None
    ref_ids = None
    if ref_images:
        with torch.no_grad():
            ref_tokens, ref_ids = encode_image_refs(ae, ref_images)

    return KleinPipeline(
        model_name=model_name,
        model_info=model_info,
        device=device,
        text_encoder=text_encoder,
        model=model,
        ae=ae,
        num_steps=defaults.get("num_steps", 50),
        guidance=defaults.get("guidance", 4.0),
        width=width,
        height=height,
        ref_tokens=ref_tokens,
        ref_ids=ref_ids,
    )


@torch.no_grad()
def generate_image(pipe: KleinPipeline, prompt: str, seed: int) -> Image.Image:
    text_encoder = pipe.text_encoder
    model = pipe.model

    if pipe.model_info["guidance_distilled"]:
        ctx = text_encoder([prompt]).to(torch.bfloat16)
    else:
        ctx_empty = text_encoder([""]).to(torch.bfloat16)
        ctx_prompt = text_encoder([prompt]).to(torch.bfloat16)
        ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
    ctx, ctx_ids = batched_prc_txt(ctx)

    shape = (1, 128, pipe.height // 16, pipe.width // 16)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device=pipe.device)
    x, x_ids = batched_prc_img(randn)

    timesteps = get_schedule(pipe.num_steps, x.shape[1])
    if pipe.model_info["guidance_distilled"]:
        denoise_fn = (
            denoise_cached
            if (pipe.model_info.get("use_kv_cache") and pipe.ref_tokens is not None)
            else denoise
        )
        x = denoise_fn(
            model,
            x,
            x_ids,
            ctx,
            ctx_ids,
            timesteps=timesteps,
            guidance=pipe.guidance,
            img_cond_seq=pipe.ref_tokens,
            img_cond_seq_ids=pipe.ref_ids,
        )
    else:
        x = denoise_cfg(
            model,
            x,
            x_ids,
            ctx,
            ctx_ids,
            timesteps=timesteps,
            guidance=pipe.guidance,
            img_cond_seq=pipe.ref_tokens,
            img_cond_seq_ids=pipe.ref_ids,
        )

    x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
    x = pipe.ae.decode(x).float()
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    return Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
