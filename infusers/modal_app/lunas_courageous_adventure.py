"""Klein 9B on Modal — Volume-backed weights, reqm quant chokepoint."""

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import modal
import torch

APP_NAME = "lunas-courageous-adventure"
VOLUME_NAME = "jkvc-klein-9b-weights"
WEIGHTS_MOUNT = Path("/weights")
HF_HOME = WEIGHTS_MOUNT / "klein-9b" / "hf"
DEFAULT_RECIPE = "quant/flux/klein9b/image_basic"

weights_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "reqm>=0.1",
        "torch==2.8.0",
        "torchvision",
        "transformers==4.56.1",
        "einops==0.8.1",
        "safetensors==0.4.5",
        "pillow>=10",
        "sentencepiece>=0.2.0",
        "fastapi[standard]>=0.115",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("git+https://github.com/black-forest-labs/flux2.git")
    .add_local_python_source("infusers")
    # add_local_python_source ships .py only; reqm YAML lives in infusers/configs/
    .add_local_dir(
        Path(__file__).resolve().parent.parent / "configs",
        remote_path="/root/infusers/configs",
    )
)

app = modal.App(APP_NAME, image=image)


def _decode_cond_images_base64(
    encoded: list[str] | None,
    device: torch.device,
) -> list[torch.Tensor] | None:
    if not encoded:
        return None
    from PIL import Image

    from infusers.quant.api.image_base import pil_to_chw_float01

    tensors: list[torch.Tensor] = []
    for item in encoded:
        raw = base64.b64decode(item)
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        tensors.append(pil_to_chw_float01(pil, device))
    return tensors


@app.cls(
    gpu="L40S",
    volumes={str(WEIGHTS_MOUNT): weights_vol},
    scaledown_window=120,
    timeout=600,
)
class LunasCourageousAdventure:
    @modal.enter()
    def setup(self) -> None:
        from infusers import QM

        t0 = time.perf_counter()
        os.environ["HF_HOME"] = str(HF_HOME)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        ckpt = WEIGHTS_MOUNT / "klein-9b" / "klein-9b"
        if not ckpt.is_dir():
            raise FileNotFoundError(
                f"Missing weights at {ckpt}. Run ./scripts/upload_weights.sh from your machine."
            )

        self.quant = QM.build(DEFAULT_RECIPE)
        print(f"Quant ready ({DEFAULT_RECIPE}) in {time.perf_counter() - t0:.1f}s")

    @modal.method()
    def infer(
        self,
        prompt: str,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        cond_images_base64: list[str] | None = None,
    ) -> bytes:
        from infusers.quant.api.image_base import chw_float01_to_pil

        device = self.quant.model.device
        cond = _decode_cond_images_base64(cond_images_base64, device)
        resolution = [height, width] if height is not None and width is not None else None
        out = self.quant(
            prompt=prompt,
            seed=seed,
            resolution=resolution,
            cond_images=cond,
        )
        pil = chw_float01_to_pil(out.image)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    @modal.fastapi_endpoint(method="POST", docs=True)
    def web(self, item: dict):
        from fastapi.responses import Response

        prompt = str(item.get("prompt", "A cat holding a sign that says hello world"))
        seed_raw = item.get("seed")
        seed = int(seed_raw) if seed_raw is not None else None
        width_raw = item.get("width")
        height_raw = item.get("height")
        width = int(width_raw) if width_raw is not None else None
        height = int(height_raw) if height_raw is not None else None
        cond_b64 = item.get("cond_images_base64")
        cond_list = list(cond_b64) if cond_b64 else None
        jpeg = self.infer.local(
            prompt,
            seed=seed,
            width=width,
            height=height,
            cond_images_base64=cond_list,
        )
        return Response(content=jpeg, media_type="image/jpeg")


@app.local_entrypoint()
def smoke(prompt: str = "solid red square on white background", seed: int = 42) -> None:
    """CLI smoke: uv run modal run infusers/modal_app/lunas_courageous_adventure.py::smoke"""
    service = LunasCourageousAdventure()
    t0 = time.perf_counter()
    jpeg = service.infer.remote(prompt, seed=seed, width=512, height=512)
    elapsed = time.perf_counter() - t0
    out = Path("/tmp/klein-smoke.jpg")
    out.write_bytes(jpeg)
    print(f"done in {elapsed:.1f}s -> {out} ({len(jpeg)} bytes)")
