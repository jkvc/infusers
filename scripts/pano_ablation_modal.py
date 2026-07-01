#!/usr/bin/env python3
"""Run immersive-room-style pano ablations against the deployed Modal endpoint."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

ATHENS_PROMPT = (
    "A full-bleed panorama of a sun-washed Athenian street, in the visual language of "
    "a cinematic travel-documentary establishing shot — tabby, cream, and grey cats "
    "scattered across weathered terracotta tiles and draped over whitewashed stone steps, "
    "clustered around overturned ceramic bowls in the left foreground, lounging beneath "
    "bougainvillea vines cascading magenta across crumbling ochre walls, and silhouetted "
    "against peeling blue wooden shutters in the central and right thirds. Overhead "
    "Mediterranean light rakes warm and golden from the upper left, casting sharp shadows "
    "of grapevines and architectural details across sun-bleached plaster. Color palette: "
    "dusty cream, warm terracotta, and faded Aegean blue — no other colors anywhere in the "
    "scene; cinematic photography with warm film grain and shallow depth of field, "
    "capturing the languid heat and feline indifference of urban Athens. No text or "
    "letterforms anywhere in the scene. Ultra-high resolution, full bleed, edge to edge."
)

SPLIT_LEFT = (
    "A full-bleed panorama left third of a sun-washed Athenian street — tabby and cream cats "
    "around overturned ceramic bowls on weathered terracotta tiles, bougainvillea magenta on "
    "ochre walls, dusty cream and warm terracotta palette, cinematic travel photography. "
    "No text or letterforms anywhere in the scene. Ultra-high resolution, full bleed, edge to edge."
)

SPLIT_RIGHT = (
    "A full-bleed panorama right third of a sun-washed Athenian street — grey cats silhouetted "
    "against peeling Aegean-blue shutters and whitewashed stone steps, golden Mediterranean "
    "light from upper left, faded Aegean blue and warm terracotta palette, cinematic travel "
    "photography. No text or letterforms anywhere in the scene. Ultra-high resolution, full "
    "bleed, edge to edge."
)

# Cabin: 1536 * 0.36 ≈ 553; snap to ÷16 for infusers overlap_pixels validation.
CABIN_OVERLAP = 544

CASES: list[dict[str, object]] = [
    {
        "name": "cabin_12s_2slice_same_768x1536",
        "prompts": [ATHENS_PROMPT, ATHENS_PROMPT],
        "resolution": [768, 1536],
        "overlap_pixels": CABIN_OVERLAP,
        "num_steps": 12,
    },
    {
        "name": "infusers_default_4s_512x1024",
        "prompts": [ATHENS_PROMPT, ATHENS_PROMPT],
        "resolution": [512, 1024],
        "overlap_pixels": 256,
        "num_steps": 4,
    },
    {
        "name": "cabin_4s_2slice_same_768x1536",
        "prompts": [ATHENS_PROMPT, ATHENS_PROMPT],
        "resolution": [768, 1536],
        "overlap_pixels": CABIN_OVERLAP,
        "num_steps": 4,
    },
    {
        "name": "cabin_12s_3slice_same_768x1536",
        "prompts": [ATHENS_PROMPT, ATHENS_PROMPT, ATHENS_PROMPT],
        "resolution": [768, 1536],
        "overlap_pixels": CABIN_OVERLAP,
        "num_steps": 12,
    },
    {
        "name": "cabin_12s_2slice_split_prompts",
        "prompts": [SPLIT_LEFT, SPLIT_RIGHT],
        "resolution": [768, 1536],
        "overlap_pixels": CABIN_OVERLAP,
        "num_steps": 12,
    },
    {
        "name": "cabin_12s_overlap256_768x1536",
        "prompts": [ATHENS_PROMPT, ATHENS_PROMPT],
        "resolution": [768, 1536],
        "overlap_pixels": 256,
        "num_steps": 12,
    },
    {
        "name": "cabin_8s_2slice_same_768x1536",
        "prompts": [ATHENS_PROMPT, ATHENS_PROMPT],
        "resolution": [768, 1536],
        "overlap_pixels": CABIN_OVERLAP,
        "num_steps": 8,
    },
]


def _load_env(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _post_pano(url: str, headers: dict[str, str], body: dict[str, object]) -> dict[str, object]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    _load_env(repo_root)

    url = os.environ.get("MODAL_WEB_URL")
    key = os.environ.get("MODAL_KEY")
    secret = os.environ.get("MODAL_SECRET")
    if not url or not key or not secret:
        print("Set MODAL_WEB_URL, MODAL_KEY, MODAL_SECRET in .env", file=sys.stderr)
        return 1

    out_dir = repo_root / ".model-out" / "pano-ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = {"Modal-Key": key, "Modal-Secret": secret}
    seed = 42
    results: list[dict[str, object]] = []

    for case in CASES:
        name = str(case["name"])
        inputs: dict[str, object] = {
            "prompts": case["prompts"],
            "seed": seed,
            "resolution": case["resolution"],
            "pano_direction": "horizontal",
            "overlap_pixels": case["overlap_pixels"],
            "num_steps": case["num_steps"],
        }
        body = {"path": "klein9b.pano", "inputs": inputs}
        print(f"Running {name} ...", flush=True)
        t0 = time.perf_counter()
        try:
            response = _post_pano(url, headers, body)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"  FAIL HTTP {exc.code}: {detail[:500]}", file=sys.stderr)
            results.append({"name": name, "ok": False, "error": detail[:500]})
            continue
        except urllib.error.URLError as exc:
            print(f"  FAIL: {exc}", file=sys.stderr)
            results.append({"name": name, "ok": False, "error": str(exc)})
            continue

        elapsed = time.perf_counter() - t0
        image_b64_list = response.get("result", {}).get("images")
        if not image_b64_list:
            results.append({"name": name, "ok": False, "error": "no images in response"})
            continue

        image_b64 = image_b64_list[0]
        raw = base64.b64decode(image_b64)
        webp_path = out_dir / f"{name}.webp"
        webp_path.write_bytes(raw)
        img = Image.open(io.BytesIO(raw))
        meta = response.get("metadata", {})
        entry = {
            "name": name,
            "ok": True,
            "elapsed_s": round(elapsed, 2),
            "webp": str(webp_path.relative_to(repo_root)),
            "output_size": [img.height, img.width],
            "num_slices": len(case["prompts"]),
            "inputs": {
                k: v for k, v in inputs.items() if k != "prompts"
            },
            "metadata": meta,
        }
        results.append(entry)
        print(
            f"  ok {elapsed:.1f}s -> {webp_path.name} "
            f"({img.width}x{img.height}, {len(case['prompts'])} slices)",
            flush=True,
        )

    summary_path = out_dir / "results.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    ok_count = sum(1 for r in results if r.get("ok"))
    print(f"\n{ok_count}/{len(results)} passed -> {summary_path}")
    return 0 if ok_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
