"""Unit tests for generator quant archetypes."""

from __future__ import annotations

import pytest
import torch
from reqm.overrides_ext import override

from infusers import QM
from infusers.quant.api.base import FinalEvent, IntermediateEvent, TorchQuant
from infusers.quant.api.image_base import (
    DummyImageQuant,
    ImageIntermediateEvent,
    ImageOutput,
    ImageQuant,
)


def test_intermediate_event_is_frozen() -> None:
    event = IntermediateEvent(message="hello")
    assert event.message == "hello"
    with pytest.raises(AttributeError):
        event.message = "other"  # type: ignore[misc]


def test_image_intermediate_event_is_subclass() -> None:
    event = ImageIntermediateEvent(message="img progress")
    assert isinstance(event, IntermediateEvent)
    assert event.message == "img progress"


def test_image_output_extends_final_event() -> None:
    out = ImageOutput(message="done", image=torch.zeros(3, 2, 2))
    assert isinstance(out, FinalEvent)
    assert out.message == "done"


def test_dummy_forward_gen_yields_progress_then_image() -> None:
    quant = DummyImageQuant(num_steps=2, resolution=[8, 8])
    events: list[ImageIntermediateEvent | ImageOutput] = list(
        quant.forward_gen(prompt="test", seed=1)
    )

    assert len(events) == 4  # begin + 2 steps + final
    assert all(isinstance(e, ImageIntermediateEvent) for e in events[:-1])
    assert events[0].message == "dummy: begin"
    assert events[1].message == "dummy: step 1/2"
    assert events[2].message == "dummy: step 2/2"
    assert isinstance(events[-1], ImageOutput)
    assert events[-1].message == "dummy: done"
    assert events[-1].image.shape == (3, 8, 8)


def test_dummy_forward_drains_forward_gen() -> None:
    quant = DummyImageQuant(num_steps=1, resolution=[4, 4])
    out = quant(prompt="drain", seed=99)
    assert isinstance(out, ImageOutput)
    assert out.image.shape == (3, 4, 4)


def test_dummy_forward_is_deterministic_with_seed() -> None:
    a = DummyImageQuant(num_steps=1, resolution=[2, 2])(prompt="x", seed=5)
    b = DummyImageQuant(num_steps=1, resolution=[2, 2])(prompt="y", seed=5)
    assert torch.allclose(a.image, b.image)


def test_forward_gen_without_final_raises_on_forward() -> None:
    class EmptyGen(TorchQuant[IntermediateEvent, str]):
        @override
        def forward_gen(self, **kwargs: object):
            yield IntermediateEvent(message="only progress")

        @override
        def dummy_inputs(self) -> list[dict[str, object]]:
            return [{}]

    with pytest.raises(RuntimeError, match="did not yield a final result"):
        EmptyGen()()


def test_image_quant_mro() -> None:
    assert issubclass(DummyImageQuant, ImageQuant)
    assert issubclass(ImageQuant, TorchQuant)


def test_qm_builds_dummy_recipe() -> None:
    quant = QM.build("quant/image_basic_dummy")
    assert isinstance(quant, DummyImageQuant)
    out = quant(prompt="via reqm", seed=0)
    assert out.image.shape[-2:] == (64, 64)
