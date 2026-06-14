#!/usr/bin/env python3
"""Klein t2i via BFL flux2 — load once, infer many prompts.

Requires --output-dir and at least one --prompt (repeat -p for a batch).

Example:

    uv run python scripts/vanilla_inference_klein.py \\
      -o .model-out/klein-smoke \\
      -p "a cat holding a sign that says hello world" \\
      -p "a red panda in a bamboo forest, soft morning light"
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import click
import torch
import torch.nn as nn
from einops import rearrange
from flux2.autoencoder import AutoEncoder
from flux2.model import Flux2
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
from flux2.text_encoder import Qwen3Embedder
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder
from PIL import Image

KLEIN_MODELS = [name for name in FLUX2_MODEL_INFO if "klein" in name]
_FP8_MIN_CAPABILITY = (8, 9)
_AE_REPO = "black-forest-labs/FLUX.2-dev"


def _check_hf_access() -> None:
    if os.environ.get("HF_TOKEN") or Path.home().joinpath(".cache/huggingface/token").exists():
        return
    click.echo(
        f"Note: VAE weights come from gated repo {_AE_REPO}.\n"
        "Run `uv run hf auth login` after accepting the FLUX license on Hugging Face.",
        err=True,
    )


def _qwen_variant_for_model(model_name: str) -> str:
    name = model_name.lower()
    if "4b" in name:
        return "4B"
    if "9b" in name:
        return "8B"
    raise click.ClickException(f"No Qwen3 fallback mapping for model {model_name!r}")


def load_text_encoder_compat(model_name: str, device: torch.device) -> Qwen3Embedder:
    """Use BFL's loader on Ada+, bf16 Qwen3 on Ampere (e.g. RTX 3090)."""
    cap = torch.cuda.get_device_capability()
    if cap >= _FP8_MIN_CAPABILITY:
        return load_text_encoder(model_name, device=device)

    if "klein" not in model_name.lower():
        return load_text_encoder(model_name, device=device)

    variant = _qwen_variant_for_model(model_name)
    model_spec = f"Qwen/Qwen3-{variant}"
    click.echo(
        f"GPU compute capability {cap[0]}.{cap[1]} < 8.9; "
        f"using {model_spec} (bf16) instead of FP8 text encoder."
    )
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


@dataclass
class KleinPipeline:
    model_name: str
    model_info: dict
    device: torch.device
    text_encoder: Qwen3Embedder
    model: Flux2
    ae: AutoEncoder
    cpu_offload: bool
    num_steps: int
    guidance: float
    width: int
    height: int
    ref_tokens: torch.Tensor | None
    ref_ids: torch.Tensor | None


def load_pipeline(
    model_name: str,
    *,
    width: int,
    height: int,
    cpu_offload: bool,
    ref_image_paths: tuple[Path, ...] = (),
) -> KleinPipeline:
    model_name = model_name.lower()
    model_info = FLUX2_MODEL_INFO[model_name]
    defaults = model_info.get("defaults", {})
    device = torch.device("cuda")

    click.echo(
        f"Loading {model_name} "
        f"(steps={defaults.get('num_steps', 50)}, guidance={defaults.get('guidance', 4.0)})..."
    )
    text_encoder = load_text_encoder_compat(model_name, device=device)
    model = load_flow_model(model_name, device="cpu" if cpu_offload else device)
    ae = load_ae(model_name, device=device)
    model.eval()
    ae.eval()
    text_encoder.eval()

    ref_images = [Image.open(path).convert("RGB") for path in ref_image_paths]
    with torch.no_grad():
        ref_tokens, ref_ids = encode_image_refs(ae, ref_images)

    return KleinPipeline(
        model_name=model_name,
        model_info=model_info,
        device=device,
        text_encoder=text_encoder,
        model=model,
        ae=ae,
        cpu_offload=cpu_offload,
        num_steps=defaults.get("num_steps", 50),
        guidance=defaults.get("guidance", 4.0),
        width=width,
        height=height,
        ref_tokens=ref_tokens,
        ref_ids=ref_ids,
    )


def _output_name(index: int, prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:48] or "prompt"
    return f"{index:03d}-{slug}.png"


def _gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {}
    cap = torch.cuda.get_device_capability()
    return {
        "name": torch.cuda.get_device_name(0),
        "compute_capability": f"{cap[0]}.{cap[1]}",
    }


def _write_run_report(output_dir: Path, report: dict) -> Path:
    path = output_dir / "run.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def _print_timing_summary(report: dict) -> None:
    timing = report["timing"]
    click.echo("--- timing ---")
    click.echo(f"load:      {timing['load_s']:.2f}s")
    for row in timing["per_image"]:
        click.echo(f"infer [{row['index']}]: {row['inference_s']:.2f}s -> {row['output']}")
    click.echo(f"infer sum: {timing['inference_total_s']:.2f}s")
    click.echo(f"total:     {timing['total_s']:.2f}s")
    click.echo(f"report:    {report['report_path']}")


@torch.no_grad()
def generate_one(pipe: KleinPipeline, prompt: str, seed: int) -> Image.Image:
    text_encoder = pipe.text_encoder
    model = pipe.model

    if pipe.model_info["guidance_distilled"]:
        ctx = text_encoder([prompt]).to(torch.bfloat16)
    else:
        ctx_empty = text_encoder([""]).to(torch.bfloat16)
        ctx_prompt = text_encoder([prompt]).to(torch.bfloat16)
        ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
    ctx, ctx_ids = batched_prc_txt(ctx)

    if pipe.cpu_offload:
        text_encoder = text_encoder.cpu()
        torch.cuda.empty_cache()
        model = model.to(pipe.device)

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

    if pipe.cpu_offload:
        pipe.model = model.cpu()
        torch.cuda.empty_cache()
        pipe.text_encoder = text_encoder.to(pipe.device)

    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    return Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())


@click.command(
    context_settings={"max_content_width": 100},
    epilog=(
        "Example:\n"
        "  uv run python scripts/vanilla_inference_klein.py \\\n"
        "    -o .model-out/klein-smoke \\\n"
        '    -p "a cat holding a sign that says hello world" \\\n'
        '    -p "a red panda in a bamboo forest, soft morning light"'
    ),
)
@click.option(
    "--prompt",
    "-p",
    "prompts",
    multiple=True,
    required=True,
    help="Prompt(s). Required; pass -p once per image.",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path, file_okay=False),
    help="Output directory. Required; e.g. .model-out/<experiment-name>/",
)
@click.option(
    "--model",
    "-m",
    "model_name",
    default="flux.2-klein-4b",
    show_default=True,
    type=click.Choice(sorted(FLUX2_MODEL_INFO.keys()), case_sensitive=False),
    help=f"Model name. Klein variants: {', '.join(KLEIN_MODELS)}.",
)
@click.option(
    "--seed",
    default=0,
    show_default=True,
    help="Base random seed; each prompt uses seed + index.",
)
@click.option("--width", default=1024, show_default=True, help="Output width in pixels.")
@click.option("--height", default=1024, show_default=True, help="Output height in pixels.")
@click.option(
    "--cpu-offload",
    is_flag=True,
    help="Load flow model on CPU and move to GPU per inference (helps tight VRAM).",
)
@click.option(
    "--ref-image",
    "-r",
    "ref_images",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Reference image(s) for i2i / editing. Pass -r once per image.",
)
def main(
    prompts: tuple[str, ...],
    output_dir: Path,
    model_name: str,
    seed: int,
    width: int,
    height: int,
    cpu_offload: bool,
    ref_images: tuple[Path, ...],
) -> None:
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required for Klein inference.")

    if not prompts:
        raise click.ClickException("At least one --prompt / -p is required.")

    _check_hf_access()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_started = datetime.now(UTC)
    t0 = time.perf_counter()

    load_start = time.perf_counter()
    pipe = load_pipeline(
        model_name,
        width=width,
        height=height,
        cpu_offload=cpu_offload,
        ref_image_paths=ref_images,
    )
    load_s = time.perf_counter() - load_start
    click.echo(f"load: {load_s:.2f}s")

    per_image: list[dict] = []
    for index, prompt in enumerate(prompts):
        infer_start = time.perf_counter()
        image = generate_one(pipe, prompt, seed=seed + index)
        out_name = _output_name(index, prompt)
        out_path = output_dir / out_name
        image.save(out_path)
        infer_s = time.perf_counter() - infer_start
        row = {
            "index": index,
            "prompt": prompt,
            "seed": seed + index,
            "output": out_name,
            "inference_s": round(infer_s, 3),
        }
        per_image.append(row)
        click.echo(f"infer [{index}]: {infer_s:.2f}s -> {out_path}")

    total_s = time.perf_counter() - t0
    inference_total_s = sum(row["inference_s"] for row in per_image)
    run_finished = datetime.now(UTC)
    report_path = output_dir / "run.json"

    report = {
        "started_at": run_started.isoformat(),
        "finished_at": run_finished.isoformat(),
        "report_path": str(report_path),
        "config": {
            "model": pipe.model_name,
            "num_steps": pipe.num_steps,
            "guidance": pipe.guidance,
            "width": width,
            "height": height,
            "seed_base": seed,
            "cpu_offload": cpu_offload,
            "ref_images": [str(p.resolve()) for p in ref_images],
            "prompt_count": len(prompts),
            "output_dir": str(output_dir.resolve()),
            "gpu": _gpu_info(),
        },
        "timing": {
            "load_s": round(load_s, 3),
            "inference_total_s": round(inference_total_s, 3),
            "total_s": round(total_s, 3),
            "per_image": per_image,
        },
        "outputs": [row["output"] for row in per_image],
    }
    _write_run_report(output_dir, report)
    _print_timing_summary(report)


if __name__ == "__main__":
    main()
