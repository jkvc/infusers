"""Klein 9B on Modal — generic runner with Volume-backed weights."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

import modal

from infusers.modal_app.base import GenericModelRunner, OutputMapping, RouteDef, RunnerError
from infusers.modal_app.translators.atomic import NCHWToWebpB64List, TensorToWebpB64
from infusers.model.weights import FLOW_FILENAME

APP_NAME = "lunas-courageous-adventure"
STREAM_LABEL = f"{APP_NAME}-stream"
VOLUME_NAME = "jkvc-klein-9b-weights"
WEIGHTS_MOUNT = Path("/weights")
HF_HOME = WEIGHTS_MOUNT / "klein-9b" / "hf"

weights_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "reqm>=0.1.1",
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
    .add_local_dir(
        Path(__file__).resolve().parent.parent / "configs",
        remote_path="/root/infusers/configs",
    )
)

app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu="L40S",
    volumes={str(WEIGHTS_MOUNT): weights_vol},
    scaledown_window=300,
    timeout=600,
)
class LunasCourageousAdventure(GenericModelRunner):
    ROUTES = [
        RouteDef(
            path="klein9b.image",
            recipe="quant/flux/klein9b/image_basic",
            intermediate_outputs=[
                OutputMapping(consume_from="message", produce_to="message"),
            ],
            final_outputs=[
                OutputMapping(
                    consume_from="image",
                    produce_to="image",
                    translators=[TensorToWebpB64()],
                ),
            ],
            allowed_input_translators={
                "cond_images": ["list_apply[imageb64_to_tensor]"],
            },
        ),
        RouteDef(
            path="klein9b.pano",
            recipe="quant/flux/klein9b/pano_basic",
            intermediate_outputs=[
                OutputMapping(consume_from="message", produce_to="message"),
            ],
            final_outputs=[
                OutputMapping(
                    consume_from="images",
                    produce_to="images",
                    translators=[NCHWToWebpB64List()],
                ),
                OutputMapping(consume_from="direction", produce_to="direction"),
                OutputMapping(consume_from="slice_resolution", produce_to="slice_resolution"),
                OutputMapping(consume_from="output_size", produce_to="output_size"),
                OutputMapping(consume_from="num_slices", produce_to="num_slices"),
                OutputMapping(consume_from="overlap_pixels", produce_to="overlap_pixels"),
            ],
            allowed_input_translators={
                "cond_images": ["list_apply[imageb64_to_tensor]"],
            },
        ),
    ]

    @modal.enter()
    def setup(self) -> None:
        t0 = time.perf_counter()
        os.environ["HF_HOME"] = str(HF_HOME)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        ckpt = WEIGHTS_MOUNT / "klein-9b" / "klein-9b"
        flow = ckpt / FLOW_FILENAME
        if not flow.is_file():
            raise FileNotFoundError(
                f"Missing weights at {flow}. "
                "Run ./scripts/stage_weights.sh && ./scripts/upload_weights.sh from your machine."
            )

        self.init_routes()
        self.get_quant(self.ROUTES[0].recipe)
        print(f"Runner ready ({self.ROUTES[0].path}) in {time.perf_counter() - t0:.1f}s")

    @modal.method()
    def run_remote(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.run(body)

    @modal.method()
    def run_stream_remote(self, body: dict[str, Any]) -> list[str]:
        return list(self.run_stream(body))

    @modal.fastapi_endpoint(method="POST", docs=True, label=APP_NAME, requires_proxy_auth=True)
    def web(self, item: dict[str, Any]):
        from fastapi import HTTPException
        from fastapi.responses import JSONResponse

        try:
            return JSONResponse(self.run(item))
        except RunnerError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @modal.fastapi_endpoint(method="POST", docs=True, label=STREAM_LABEL, requires_proxy_auth=True)
    def web_stream(self, item: dict[str, Any]):
        from fastapi import HTTPException
        from fastapi.responses import StreamingResponse

        try:
            return StreamingResponse(self.run_stream(item), media_type="text/event-stream")
        except RunnerError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.local_entrypoint()
def smoke(
    prompt: str = "solid red square on white background",
    seed: int = 42,
    num_steps: int | None = None,
) -> None:
    """CLI smoke: uv run modal run infusers/modal_app/lunas_courageous_adventure.py::smoke"""
    service = LunasCourageousAdventure()
    t0 = time.perf_counter()

    inputs: dict[str, object] = {"prompt": prompt, "seed": seed, "resolution": [512, 512]}
    if num_steps is not None:
        inputs["num_steps"] = num_steps

    body = {
        "path": "klein9b.image",
        "inputs": inputs,
    }
    response = service.run_remote.remote(body)
    elapsed = time.perf_counter() - t0

    image_b64 = response["result"]["image"]
    step_label = num_steps if num_steps is not None else "default"
    out = Path(f"/tmp/klein-smoke-steps-{step_label}.webp")
    out.write_bytes(base64.b64decode(image_b64))
    meta = response.get("metadata", {})
    print(f"done in {elapsed:.1f}s -> {out} ({out.stat().st_size} bytes)")
    print(f"metadata: {meta}")


@app.local_entrypoint()
def smoke_cond(
    prompt: str = "recreate this exact solid red square",
    seed: int = 42,
) -> None:
    """CLI smoke with cond_images: uv run modal run ...::smoke_cond"""
    import io

    from PIL import Image

    service = LunasCourageousAdventure()
    t0 = time.perf_counter()

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(255, 0, 0)).save(buf, format="PNG")
    cond_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    body = {
        "path": "klein9b.image",
        "inputs": {
            "prompt": prompt,
            "seed": seed,
            "resolution": [512, 512],
            "cond_images": [cond_b64],
        },
        "translator": {"cond_images": "list_apply[imageb64_to_tensor]"},
    }
    response = service.run_remote.remote(body)
    elapsed = time.perf_counter() - t0

    image_b64 = response["result"]["image"]
    out = Path("/tmp/klein-smoke-cond.webp")
    out.write_bytes(base64.b64decode(image_b64))
    meta = response.get("metadata", {})
    print(f"done in {elapsed:.1f}s -> {out} ({out.stat().st_size} bytes)")
    print(f"metadata: {meta}")


@app.local_entrypoint()
def smoke_pano(
    seed: int = 42,
    num_steps: int | None = None,
) -> None:
    """CLI pano smoke: modal run .../lunas_courageous_adventure.py::smoke_pano"""
    service = LunasCourageousAdventure()
    t0 = time.perf_counter()

    inputs: dict[str, object] = {
        "prompts": ["warm desert dunes at sunset", "cool ocean horizon at dusk"],
        "seed": seed,
        "resolution": [512, 1024],
        "pano_direction": "horizontal",
        "overlap_pixels": 256,
    }
    if num_steps is not None:
        inputs["num_steps"] = num_steps

    body = {
        "path": "klein9b.pano",
        "inputs": inputs,
    }
    response = service.run_remote.remote(body)
    elapsed = time.perf_counter() - t0

    images_b64 = response["result"]["images"]
    if not images_b64:
        raise RuntimeError("smoke_pano: empty result.images")
    step_label = num_steps if num_steps is not None else "default"
    out = Path(f"/tmp/klein-smoke-pano-steps-{step_label}.webp")
    out.write_bytes(base64.b64decode(images_b64[0]))
    meta = response.get("metadata", {})
    result = response.get("result", {})
    print(
        f"done in {elapsed:.1f}s -> {out} ({out.stat().st_size} bytes, "
        f"{len(images_b64)} image(s))"
    )
    print(
        f"output: {result.get('output_size')} direction={result.get('direction')} "
        f"slices={result.get('num_slices')} overlap={result.get('overlap_pixels')}"
    )
    print(f"metadata: {meta}")


@app.local_entrypoint()
def smoke_describe() -> None:
    """Describe smoke: uv run modal run ...::smoke_describe"""
    service = LunasCourageousAdventure()
    response = service.run_remote.remote({"path": "__DESCRIBE__"})
    routes = response["result"]["routes"]
    print(f"routes: {list(routes.keys())}")
    print(f"translators: {response['result']['available_translators']}")
