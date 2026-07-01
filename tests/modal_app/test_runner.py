"""Tests for GenericModelRunner dispatch and validation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
import torch

from infusers.modal_app.base import (
    DESCRIBE_PATH,
    GenericModelRunner,
    OutputMapping,
    RouteDef,
    RunnerError,
)
from infusers.modal_app.translators.atomic import TensorToWebpB64
from infusers.quant.api.base import IntermediateEvent
from infusers.quant.api.image_base import ImageOutput


@dataclass
class EchoOutput:
    echo: str
    image: torch.Tensor


class MockQuant:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.model = self
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        tensor = torch.zeros(3, 2, 2)
        return EchoOutput(echo=kwargs.get("prompt", ""), image=tensor)


class MockStreamQuant:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.model = self
        self.calls: list[dict] = []

    def forward_gen(self, **kwargs) -> Iterator[IntermediateEvent | ImageOutput]:
        self.calls.append(kwargs)
        yield IntermediateEvent(message="mock step 1")
        yield IntermediateEvent(message="mock step 2")
        yield ImageOutput(message="mock done", image=torch.ones(3, 2, 2))

    def __call__(self, **kwargs):
        final = None
        for item in self.forward_gen(**kwargs):
            if isinstance(item, IntermediateEvent):
                continue
            final = item
        return final


ECHO_ROUTE = RouteDef(
    path="test/echo",
    recipe="quant/test/echo",
    intermediate_outputs=[
        OutputMapping(consume_from="message", produce_to="message"),
    ],
    final_outputs=[
        OutputMapping(consume_from="image", produce_to="image", translators=[TensorToWebpB64()]),
        OutputMapping(consume_from="echo", produce_to="echo"),
    ],
    allowed_input_translators={
        "tags": ["list_apply[identity]"],
    },
)


class EchoRunner(GenericModelRunner):
    ROUTES = [ECHO_ROUTE]


@pytest.fixture
def runner() -> EchoRunner:
    r = EchoRunner()
    r.init_routes()
    mock = MockQuant()
    r._quants[ECHO_ROUTE.recipe] = mock
    r._mock = mock  # type: ignore[attr-defined]
    return r


def test_describe_returns_routes_and_translators(runner: EchoRunner) -> None:
    out = runner.run({"path": DESCRIBE_PATH})
    assert "result" in out
    route_desc = out["result"]["routes"][ECHO_ROUTE.path]
    assert route_desc["final_outputs"][0]["translators"] == ["TensorToWebpB64()"]
    assert route_desc["output_schema"] == {
        "image": "string (webp base64)",
        "echo": "pass-through",
    }
    assert route_desc["stream_schema"]["progress"] == {"message": "pass-through"}
    assert "imageb64_to_tensor" in out["result"]["available_translators"]
    assert out["metadata"]["path"] == DESCRIBE_PATH


def test_infer_happy_path(runner: EchoRunner) -> None:
    out = runner.run(
        {
            "path": ECHO_ROUTE.path,
            "inputs": {"prompt": "hello"},
        }
    )
    assert "image" in out["result"]
    assert isinstance(out["result"]["image"], str)
    assert out["result"]["echo"] == "hello"
    assert out["metadata"]["path"] == ECHO_ROUTE.path
    assert runner._mock.calls == [{"prompt": "hello"}]  # type: ignore[attr-defined]


def test_infer_passes_inputs_through_without_validation(runner: EchoRunner) -> None:
    runner.run({"path": ECHO_ROUTE.path, "inputs": {}})
    assert runner._mock.calls[-1] == {}  # type: ignore[attr-defined]


def test_infer_with_input_translator(runner: EchoRunner) -> None:
    runner.run(
        {
            "path": ECHO_ROUTE.path,
            "inputs": {"prompt": "x", "tags": ["a", "b"]},
            "translator": {"tags": "list_apply[identity]"},
        }
    )
    assert runner._mock.calls[-1]["tags"] == ["a", "b"]  # type: ignore[attr-defined]


def test_missing_path_raises(runner: EchoRunner) -> None:
    with pytest.raises(RunnerError, match="path is required") as exc:
        runner.run({})
    assert exc.value.status_code == 400


def test_unknown_path_raises(runner: EchoRunner) -> None:
    with pytest.raises(RunnerError, match="Unknown path") as exc:
        runner.run({"path": "nope", "inputs": {}})
    assert exc.value.status_code == 404


def test_translator_required_when_wire_field_present(runner: EchoRunner) -> None:
    with pytest.raises(RunnerError, match="requires a translator") as exc:
        runner.run(
            {
                "path": ECHO_ROUTE.path,
                "inputs": {"prompt": "x", "tags": ["a"]},
            }
        )
    assert exc.value.status_code == 400


def test_disallowed_translator_raises(runner: EchoRunner) -> None:
    with pytest.raises(RunnerError, match="not allowed") as exc:
        runner.run(
            {
                "path": ECHO_ROUTE.path,
                "inputs": {"prompt": "x", "tags": ["a"]},
                "translator": {"tags": "identity"},
            }
        )
    assert exc.value.status_code == 400


STREAM_ROUTE = RouteDef(
    path="test/stream",
    recipe="quant/test/stream",
    intermediate_outputs=[
        OutputMapping(consume_from="message", produce_to="message"),
    ],
    final_outputs=[
        OutputMapping(consume_from="image", produce_to="image", translators=[TensorToWebpB64()]),
    ],
    allowed_input_translators={},
)


class StreamRunner(GenericModelRunner):
    ROUTES = [STREAM_ROUTE]


@pytest.fixture
def stream_runner() -> StreamRunner:
    r = StreamRunner()
    r.init_routes()
    mock = MockStreamQuant()
    r._quants[STREAM_ROUTE.recipe] = mock
    r._mock = mock  # type: ignore[attr-defined]
    return r


def test_run_stream_emits_progress_and_result(stream_runner: StreamRunner) -> None:
    body = {"path": STREAM_ROUTE.path, "inputs": {"prompt": "stream me"}}
    frames = list(stream_runner.run_stream(body))
    assert len(frames) >= 2

    progress = []
    result = None
    for frame in frames:
        payload = json.loads(frame.removeprefix("data: ").strip())
        if payload["kind"] == "progress":
            progress.append(payload["progress"]["message"])
        elif payload["kind"] == "result":
            result = payload

    assert progress == ["mock step 1", "mock step 2"]
    assert result is not None
    assert "image" in result["result"]
    assert stream_runner._mock.calls == [{"prompt": "stream me"}]  # type: ignore[attr-defined]


def test_run_stream_logs_progress_and_final(stream_runner: StreamRunner, capsys) -> None:
    body = {"path": STREAM_ROUTE.path, "inputs": {"prompt": "stream me"}}
    list(stream_runner.run_stream(body))
    out = capsys.readouterr().out

    assert '"event": "progress_event"' in out
    assert '"message": "mock step 1"' in out
    assert '"message": "mock step 2"' in out
    assert '"event": "output_translators_applied"' in out
    assert '"event": "inference_stream_end"' in out


def test_run_stream_requires_forward_gen(runner: EchoRunner) -> None:
    with pytest.raises(RunnerError, match="does not support streaming") as exc:
        list(
            runner.run_stream(
                {"path": ECHO_ROUTE.path, "inputs": {"prompt": "x"}},
            )
        )
    assert exc.value.status_code == 400
