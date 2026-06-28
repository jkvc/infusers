#!/usr/bin/env python3
"""Image inference via reqm recipe — uniform call site, no model internals."""

from __future__ import annotations

import re
import time
from pathlib import Path

import click
import torch
from PIL import Image

from infusers import QM
from infusers.quant.api.image_base import chw_float01_to_pil, pil_to_chw_float01


def _output_name(index: int, prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:48] or "prompt"
    return f"{index:03d}-{slug}.png"


@click.command(context_settings={"max_content_width": 100})
@click.option(
    "--recipe",
    required=True,
    help="reqm config name, e.g. quant/flux/klein9b/image_basic",
)
@click.option("--prompt", "-p", "prompts", multiple=True, required=True)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path, file_okay=False),
)
@click.option("--seed", default=None, type=int, help="Random seed; default random per run")
@click.option("--width", default=1024, show_default=True)
@click.option("--height", default=1024, show_default=True)
@click.option(
    "--cond",
    "cond_paths",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Conditional reference image path(s)",
)
def main(
    recipe: str,
    prompts: tuple[str, ...],
    output_dir: Path,
    seed: int | None,
    width: int,
    height: int,
    cond_paths: tuple[Path, ...],
) -> None:
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required for inference.")

    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    click.echo(f"Building recipe {recipe!r} ...")
    load_start = time.perf_counter()
    quant = QM.build(recipe)
    load_s = time.perf_counter() - load_start
    click.echo(f"load: {load_s:.2f}s")

    device = quant.model.device
    cond_tensors = None
    if cond_paths:
        cond_tensors = [
            pil_to_chw_float01(Image.open(path).convert("RGB"), device) for path in cond_paths
        ]

    for index, prompt in enumerate(prompts):
        infer_start = time.perf_counter()
        out = quant(
            prompt=prompt,
            seed=seed,
            resolution=[height, width],
            cond_images=cond_tensors,
        )
        pil = chw_float01_to_pil(out.image)
        out_name = _output_name(index, prompt)
        out_path = output_dir / out_name
        pil.save(out_path)
        infer_s = time.perf_counter() - infer_start
        click.echo(f"infer [{index}]: {infer_s:.2f}s -> {out_path}")

    click.echo(f"total: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
