#!/usr/bin/env python3
"""Panorama inference via reqm recipe — one canvas from multiple slice prompts."""

from __future__ import annotations

import re
import time
from pathlib import Path

import click
import torch
from PIL import Image

from infusers import QM
from infusers.quant.api.image_base import chw_float01_to_pil, pil_to_chw_float01


def _output_name(prompts: tuple[str, ...]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", " ".join(prompts).lower()).strip("-")[:48] or "pano"
    return f"pano-{slug}.png"


@click.command(context_settings={"max_content_width": 100})
@click.option(
    "--recipe",
    required=True,
    help="reqm config name, e.g. quant/flux/klein9b/pano_basic",
)
@click.option(
    "--prompt",
    "-p",
    "prompts",
    multiple=True,
    required=True,
    help="One prompt per panorama slice (order left-to-right or top-to-bottom)",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path, file_okay=False),
)
@click.option("--seed", default=None, type=int, help="Random seed (quant default if omitted)")
@click.option(
    "--resolution",
    nargs=2,
    type=int,
    default=None,
    metavar="HEIGHT WIDTH",
    help="Per-slice size [height width] (quant default if omitted)",
)
@click.option(
    "--direction",
    type=click.Choice(["horizontal", "vertical"]),
    default=None,
    help="Panorama direction (quant default if omitted)",
)
@click.option(
    "--overlap",
    type=int,
    default=None,
    help="Overlap in apparent pixels (quant default if omitted)",
)
@click.option(
    "--num-steps",
    type=int,
    default=None,
    help="Diffusion steps override (quant default if omitted)",
)
@click.option(
    "--cond",
    "cond_paths",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Shared conditional reference image path(s) for all slices",
)
def main(
    recipe: str,
    prompts: tuple[str, ...],
    output_dir: Path,
    seed: int | None,
    resolution: tuple[int, int] | None,
    direction: str | None,
    overlap: int | None,
    num_steps: int | None,
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
    cond_tensors: list[torch.Tensor] | None = None
    if cond_paths:
        cond_tensors = [
            pil_to_chw_float01(Image.open(path).convert("RGB"), device) for path in cond_paths
        ]

    kwargs: dict[str, object] = {"prompts": list(prompts)}
    if seed is not None:
        kwargs["seed"] = seed
    if resolution is not None:
        kwargs["resolution"] = list(resolution)
    if direction is not None:
        kwargs["pano_direction"] = direction
    if overlap is not None:
        kwargs["overlap_pixels"] = overlap
    if num_steps is not None:
        kwargs["num_steps"] = num_steps
    if cond_tensors is not None:
        kwargs["cond_images"] = cond_tensors

    infer_start = time.perf_counter()
    out = quant(**kwargs)
    if out.images.shape[0] != 1:
        raise click.ClickException(
            f"Expected one panorama image (N=1), got batch size {out.images.shape[0]}"
        )
    pil = chw_float01_to_pil(out.images[0])
    out_name = _output_name(prompts)
    out_path = output_dir / out_name
    pil.save(out_path)
    infer_s = time.perf_counter() - infer_start
    click.echo(
        f"infer: {infer_s:.2f}s -> {out_path} "
        f"({out.output_size[1]}x{out.output_size[0]}, {out.num_slices} slices)"
    )
    click.echo(f"total: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
