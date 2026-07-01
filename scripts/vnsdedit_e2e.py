#!/usr/bin/env python3
"""E2E: t2i then localized-variation-style vnsdedit via deployed Modal HTTP API.

Writes before/after/signal_rgba to tmp/vnsdedit-e2e/.

Prereqs:
  cp .env.example .env  # MODAL_WEB_URL, MODAL_KEY, MODAL_SECRET
  uv run modal deploy infusers/modal_app/lunas_courageous_adventure.py

Usage:
  uv run python scripts/vnsdedit_e2e.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.request
from pathlib import Path

import torch
from PIL import Image

from infusers.quant.api.image_base import chw_float01_to_pil, pil_rgba_to_chw_float01
from infusers.quant.flux.vnsdedit import compose_signal_rgba, compute_radial_mask

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "tmp" / "vnsdedit-e2e"
RESOLUTION = [768, 768]
T2I_STEPS = 4
VNSD_STEPS = 20
SEED = 42

SAMPLES = [
    {
        "slug": "watercolor-street",
        "prompt": "watercolor of a quiet Flemish street, soft ochre brick, pale sky",
        "click": (420, 380),
    },
    {
        "slug": "teal-drop",
        "prompt": "single teal ink drop on white paper, minimal fluid art",
        "click": (384, 384),
    },
    {
        "slug": "pine-forest",
        "prompt": "misty pine forest at dawn, soft watercolor light",
        "click": (260, 520),
    },
]


def load_env() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        raise SystemExit("Missing .env — copy from .env.example")
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def post_json(body: dict) -> dict:
    url = os.environ["MODAL_WEB_URL"]
    key = os.environ["MODAL_KEY"]
    secret = os.environ["MODAL_SECRET"]
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Modal-Key": key,
            "Modal-Secret": secret,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())


def decode_webp_b64(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def encode_rgba_png_b64(rgb: Image.Image, mask_alpha: torch.Tensor) -> str:
    height, width = RESOLUTION
    mask = (mask_alpha[0].clamp(0, 1).numpy() * 255).astype("uint8")
    rgba = Image.merge(
        "RGBA",
        (
            rgb.getchannel("R"),
            rgb.getchannel("G"),
            rgb.getchannel("B"),
            Image.fromarray(mask, mode="L"),
        ),
    )
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def radius_fraction_to_px(radius_fraction: float, width: int, height: int) -> float:
    return max(0.0, radius_fraction) * min(width, height)


def main() -> None:
    load_env()
    OUT.mkdir(parents=True, exist_ok=True)
    height, width = RESOLUTION

    for index, sample in enumerate(SAMPLES, 1):
        slug = sample["slug"]
        print(f"[{index}/{len(SAMPLES)}] {slug}")
        t0 = time.perf_counter()

        t2i = post_json(
            {
                "path": "klein9b.image",
                "inputs": {
                    "prompt": sample["prompt"],
                    "seed": SEED,
                    "resolution": RESOLUTION,
                    "num_steps": T2I_STEPS,
                },
            }
        )
        base_pil = decode_webp_b64(t2i["result"]["image"])
        if base_pil.size != (width, height):
            raise RuntimeError(f"t2i size {base_pil.size} != {(width, height)}")
        base_path = OUT / f"{slug}-base.webp"
        base_pil.save(base_path, format="WEBP", quality=90)

        click_x, click_y = sample["click"]
        radius_px = radius_fraction_to_px(0.22, width, height)
        core_radius_px = radius_fraction_to_px(0.06, width, height)
        mask = compute_radial_mask(
            width=width,
            height=height,
            click_x=click_x,
            click_y=click_y,
            edit_max=1.0,
            radius_px=radius_px,
            core_radius_px=core_radius_px,
            falloff_shape="cosine",
        )
        base_chw = pil_rgba_to_chw_float01(base_pil.convert("RGBA"), torch.device("cpu"))[:3]
        signal_rgba = compose_signal_rgba(base_chw, mask)
        signal_pil = chw_float01_to_pil(signal_rgba[:3])
        mask_preview = Image.fromarray((mask[0].numpy() * 255).astype("uint8"), mode="L")
        signal_path = OUT / f"{slug}-signal-rgba.png"
        Image.merge(
            "RGBA",
            (
                signal_pil.getchannel("R"),
                signal_pil.getchannel("G"),
                signal_pil.getchannel("B"),
                mask_preview,
            ),
        ).save(signal_path)

        signal_b64 = encode_rgba_png_b64(base_pil, mask)
        edited = post_json(
            {
                "path": "klein9b.image",
                "inputs": {
                    "prompt": sample["prompt"],
                    "seed": SEED + index,
                    "resolution": RESOLUTION,
                    "num_steps": VNSD_STEPS,
                    "signal_rgba": signal_b64,
                },
                "translator": {"signal_rgba": "rgba_b64_to_tensor"},
            }
        )
        out_pil = decode_webp_b64(edited["result"]["image"])
        out_path = OUT / f"{slug}-vnsd.webp"
        out_pil.save(out_path, format="WEBP", quality=90)

        gallery = Image.new("RGB", (width * 3, height))
        gallery.paste(base_pil, (0, 0))
        gallery.paste(
            Image.merge(
                "RGB",
                (
                    signal_pil.getchannel("R"),
                    signal_pil.getchannel("G"),
                    signal_pil.getchannel("B"),
                ),
            ),
            (width, 0),
        )
        gallery.paste(out_pil, (width * 2, 0))
        gallery.save(OUT / f"{slug}-triptych.webp", format="WEBP", quality=90)

        elapsed = time.perf_counter() - t0
        print(
            f"  done in {elapsed:.1f}s | quant_ms={edited.get('metadata', {}).get('quant_ms')} "
            f"-> {out_path}"
        )

    print(f"\nWrote samples under {OUT}/")


if __name__ == "__main__":
    main()
