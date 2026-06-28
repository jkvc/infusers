"""Tests for GenericModelRunner dispatch and validation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from infusers.modal_app.base import DESCRIBE_PATH, GenericModelRunner, RouteDef, RunnerError
from infusers.modal_app.translators.atomic import GetAttr, TensorToWebpB64


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


ECHO_ROUTE = RouteDef(
    path="test/echo",
    recipe="quant/test/echo",
    output_key="image",
    output_translators=[GetAttr("image"), TensorToWebpB64()],
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
    assert ECHO_ROUTE.path in out["result"]["routes"]
    assert "GetAttr('image')" in out["result"]["routes"][ECHO_ROUTE.path]["output_translators"]
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
