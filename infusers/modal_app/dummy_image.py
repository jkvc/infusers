"""CPU-only Modal runner with dummy image quant — fast e2e without GPU weights."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

import modal

from infusers.modal_app.base import GenericModelRunner, OutputMapping, RouteDef, RunnerError
from infusers.modal_app.translators.atomic import TensorToWebpB64

APP_NAME = "infusers-dummy-image"
STREAM_LABEL = f"{APP_NAME}-stream"
DUMMY_PATH = "dummy.image"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "reqm>=0.1.1",
        "torch==2.8.0",
        "numpy>=2",
        "pillow>=10",
        "fastapi[standard]>=0.115",
    )
    .add_local_python_source("infusers")
    .add_local_dir(
        Path(__file__).resolve().parent.parent / "configs",
        remote_path="/root/infusers/configs",
    )
)

app = modal.App(APP_NAME, image=image)


@app.cls(scaledown_window=60, timeout=120)
class DummyImageRunner(GenericModelRunner):
    ROUTES = [
        RouteDef(
            path=DUMMY_PATH,
            recipe="quant/image_basic_dummy",
            intermediate_outputs=[
                OutputMapping(consume_from="message", produce_to="message"),
            ],
            final_outputs=[
                OutputMapping(consume_from="image", produce_to="image", translators=[TensorToWebpB64()]),
            ],
            allowed_input_translators={},
        ),
    ]

    @modal.enter()
    def setup(self) -> None:
        t0 = time.perf_counter()
        self.init_routes()
        self.get_quant(self.ROUTES[0].recipe)
        print(f"Dummy runner ready ({DUMMY_PATH}) in {time.perf_counter() - t0:.1f}s")

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
    prompt: str = "dummy smoke",
    seed: int = 42,
) -> None:
    """CLI smoke: uv run modal run infusers/modal_app/dummy_image.py::smoke"""
    service = DummyImageRunner()
    t0 = time.perf_counter()

    body = {
        "path": DUMMY_PATH,
        "inputs": {"prompt": prompt, "seed": seed},
    }
    response = service.run_remote.remote(body)
    elapsed = time.perf_counter() - t0

    image_b64 = response["result"]["image"]
    out = Path("/tmp/dummy-smoke.webp")
    out.write_bytes(base64.b64decode(image_b64))
    print(f"done in {elapsed:.1f}s -> {out} ({out.stat().st_size} bytes)")
    print(f"metadata: {response.get('metadata', {})}")


@app.local_entrypoint()
def smoke_stream(
    prompt: str = "dummy stream smoke",
    seed: int = 7,
) -> None:
    """CLI stream smoke: uv run modal run infusers/modal_app/dummy_image.py::smoke_stream"""
    service = DummyImageRunner()
    t0 = time.perf_counter()

    body = {
        "path": DUMMY_PATH,
        "inputs": {"prompt": prompt, "seed": seed},
    }
    chunks = service.run_stream_remote.remote(body)
    elapsed = time.perf_counter() - t0

    progress: list[str] = []
    result_payload: dict[str, Any] | None = None
    for chunk in chunks:
        line = chunk.removeprefix("data: ").strip()
        payload = json.loads(line)
        if payload["kind"] == "progress":
            progress.append(payload["progress"]["message"])
        elif payload["kind"] == "result":
            result_payload = payload

    assert result_payload is not None, "stream did not emit a result event"
    image_b64 = result_payload["result"]["image"]
    out = Path("/tmp/dummy-smoke-stream.webp")
    out.write_bytes(base64.b64decode(image_b64))
    print(f"progress events: {progress}")
    print(f"done in {elapsed:.1f}s -> {out} ({out.stat().st_size} bytes)")
    print(f"metadata: {result_payload.get('metadata', {})}")


@app.local_entrypoint()
def smoke_describe() -> None:
    """Describe smoke: uv run modal run infusers/modal_app/dummy_image.py::smoke_describe"""
    service = DummyImageRunner()
    response = service.run_remote.remote({"path": "__DESCRIBE__"})
    routes = response["result"]["routes"]
    print(f"routes: {list(routes.keys())}")
    print(f"translators: {response['result']['available_translators']}")
