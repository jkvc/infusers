"""Tests for streaming bridge and SSE encoding."""

from __future__ import annotations

import json
import time

import torch

from infusers.modal_app.stream import BoundedProgressBridge, encode_sse, run_generator_in_thread
from infusers.quant.api.base import IntermediateEvent
from infusers.quant.api.image_base import DummyImageQuant, ImageOutput


def test_encode_sse_format() -> None:
    payload = {"kind": "progress", "progress": {"message": "step 1/3"}}
    frame = encode_sse(payload)
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    assert json.loads(frame.removeprefix("data: ").strip()) == payload


def test_bridge_delivers_intermediates_then_final() -> None:
    bridge = BoundedProgressBridge(max_intermediate=4)
    bridge.push_intermediate(IntermediateEvent(message="a"))
    bridge.push_intermediate(IntermediateEvent(message="b"))
    bridge.push_final(ImageOutput(message="done", image=torch.zeros(3, 2, 2)))
    bridge.close()

    kinds = [frame.kind for frame in bridge.iter_frames()]
    assert kinds.count("intermediate") == 2
    assert kinds[-1] == "final"


def test_bridge_drops_intermediate_when_full() -> None:
    bridge = BoundedProgressBridge(max_intermediate=1)
    bridge.push_intermediate(IntermediateEvent(message="first"))
    bridge.push_intermediate(IntermediateEvent(message="second"))
    bridge.push_final(ImageOutput(message="done", image=torch.zeros(3, 1, 1)))
    bridge.close()

    messages = [
        frame.value.message for frame in bridge.iter_frames() if frame.kind == "intermediate"
    ]
    assert messages == ["second"]


def test_bridge_surfaces_producer_error() -> None:
    bridge = BoundedProgressBridge()

    def boom() -> None:
        bridge.push_error(ValueError("boom"))
        bridge.close()

    import threading

    threading.Thread(target=boom, daemon=True).start()
    frames = list(bridge.iter_frames())
    assert frames[-1].kind == "error"
    assert str(frames[-1].value) == "boom"


def test_run_generator_in_thread_with_dummy_quant() -> None:
    quant = DummyImageQuant(num_steps=3, resolution=[8, 8])
    bridge = BoundedProgressBridge(max_intermediate=2)
    thread = run_generator_in_thread(
        bridge,
        lambda: quant.forward_gen(prompt="thread", seed=1),
        lambda item: isinstance(item, IntermediateEvent),
    )

    progress: list[str] = []
    final: ImageOutput | None = None
    deadline = time.time() + 5
    for frame in bridge.iter_frames():
        if time.time() > deadline:
            raise AssertionError("timed out waiting for generator thread")
        if frame.kind == "intermediate":
            progress.append(frame.value.message)
        elif frame.kind == "final":
            final = frame.value
            break

    thread.join(timeout=2)
    assert progress
    assert isinstance(final, ImageOutput)
    assert final.image.shape == (3, 8, 8)
