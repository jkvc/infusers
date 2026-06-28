"""Klein 9B text-to-image on Modal — Volume-backed weights, custom flux2 stack."""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import modal

APP_NAME = "jkvc-klein-9b"
VOLUME_NAME = "jkvc-klein-9b-weights"
WEIGHTS_MOUNT = Path("/weights")
KLEIN_CKPT_DIR = WEIGHTS_MOUNT / "klein-9b" / "klein-9b"
HF_HOME = WEIGHTS_MOUNT / "klein-9b" / "hf"

weights_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
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
)

app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu="L40S",
    volumes={str(WEIGHTS_MOUNT): weights_vol},
    scaledown_window=120,  # keep GPU warm 2 min after last request
    timeout=600,
)
class Klein9B:
    @modal.enter()
    def setup(self) -> None:
        import torch

        from infusers.klein import generate_image, load_pipeline

        t0 = time.perf_counter()
        os.environ["HF_HOME"] = str(HF_HOME)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        for label, path in (("klein", KLEIN_CKPT_DIR), ("hf", HF_HOME)):
            if path.is_dir():
                sample = sorted(p.name for p in path.iterdir())[:8]
                print(f"Volume {label} ({path}): {sample}")
            else:
                raise FileNotFoundError(
                    f"Missing {label} weights at {path}. "
                    "Run ./scripts/upload_weights.sh from your machine."
                )

        print(f"Loading Klein 9B from {KLEIN_CKPT_DIR} ...")
        self.pipe = load_pipeline(
            "flux.2-klein-9b",
            width=1024,
            height=1024,
            weights_dir=KLEIN_CKPT_DIR,
        )
        _ = generate_image(self.pipe, "solid gray", seed=0)
        torch.cuda.synchronize()
        print(f"Klein 9B ready in {time.perf_counter() - t0:.1f}s")

    @modal.method()
    def infer(
        self,
        prompt: str,
        seed: int = 42,
        width: int = 1024,
        height: int = 1024,
    ) -> bytes:
        from infusers.klein import generate_image

        self.pipe.width = width
        self.pipe.height = height
        image = generate_image(self.pipe, prompt, seed=seed)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    @modal.fastapi_endpoint(method="POST", docs=True)
    def web(self, item: dict):
        from fastapi.responses import Response

        prompt = str(item.get("prompt", "A cat holding a sign that says hello world"))
        seed = int(item.get("seed", 42))
        width = int(item.get("width", 1024))
        height = int(item.get("height", 1024))
        jpeg = self.infer.local(prompt, seed=seed, width=width, height=height)
        return Response(content=jpeg, media_type="image/jpeg")


@app.local_entrypoint()
def smoke(prompt: str = "solid red square on white background", seed: int = 42) -> None:
    """CLI smoke test: uv run modal run apps/klein_9b/app.py::smoke"""
    service = Klein9B()
    t0 = time.perf_counter()
    jpeg = service.infer.remote(prompt, seed=seed, width=512, height=512)
    elapsed = time.perf_counter() - t0
    out = Path("/tmp/klein-smoke.jpg")
    out.write_bytes(jpeg)
    print(f"done in {elapsed:.1f}s -> {out} ({len(jpeg)} bytes)")
