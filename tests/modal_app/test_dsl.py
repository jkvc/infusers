"""Tests for bracket DSL parser."""

from __future__ import annotations

import pytest
import torch

from infusers.modal_app.translators.context import TranslatorContext
from infusers.modal_app.translators.dsl import parse
from infusers.modal_app.translators.registry import registered_names


def test_registered_names_include_core_translators() -> None:
    names = registered_names()
    for expected in (
        "imageb64_to_tensor",
        "tensor_to_webp_b64",
        "nchw_to_webp_b64_list",
        "get_attr",
        "list_apply",
        "pipe",
    ):
        assert expected in names


def test_parse_atomic() -> None:
    fn = parse("identity")
    assert fn("hello", TranslatorContext()) == "hello"


def test_parse_get_attr() -> None:
    fn = parse("get_attr('image')")
    out = fn(type("Out", (), {"image": 42})(), TranslatorContext())
    assert out == 42


def test_parse_list_apply() -> None:
    fn = parse("list_apply[identity]")
    assert fn([1, 2, 3], TranslatorContext()) == [1, 2, 3]


def test_parse_pipe() -> None:
    fn = parse("pipe[get_attr('x'), identity]")
    obj = {"x": "value"}
    assert fn(obj, TranslatorContext()) == "value"


def test_parse_nested_pipe_and_list_apply() -> None:
    dsl = "pipe[get_attr('items'), list_apply[identity]]"
    fn = parse(dsl)
    obj = {"items": ["a", "b"]}
    assert fn(obj, TranslatorContext()) == ["a", "b"]


def test_apply_klein_output_chain_cpu() -> None:
    from dataclasses import dataclass

    from infusers.modal_app.translators.atomic import GetAttr, TensorToWebpB64
    from infusers.modal_app.translators.registry import apply_chain

    @dataclass
    class FakeOut:
        image: torch.Tensor

    tensor = torch.zeros(3, 4, 4)
    tensor[0, :, :] = 1.0  # red
    fake = FakeOut(image=tensor)

    b64 = apply_chain([GetAttr("image"), TensorToWebpB64()], fake, TranslatorContext())
    assert isinstance(b64, str)
    assert len(b64) > 0


def test_apply_pano_nchw_output_chain_cpu() -> None:
    from infusers.modal_app.translators.atomic import NCHWToWebpB64List
    from infusers.modal_app.translators.registry import apply_chain

    images = torch.zeros(2, 3, 4, 4)
    images[0, 0, :, :] = 1.0
    images[1, 2, :, :] = 1.0

    b64_list = apply_chain([NCHWToWebpB64List()], images, TranslatorContext())
    assert isinstance(b64_list, list)
    assert len(b64_list) == 2
    assert all(isinstance(item, str) and len(item) > 0 for item in b64_list)


def test_parse_invalid_atom_raises() -> None:
    with pytest.raises(ValueError, match="Invalid translator atom"):
        parse("not a valid!!!")


def test_parse_unknown_translator_raises() -> None:
    with pytest.raises(KeyError, match="Unknown translator"):
        parse("does_not_exist")
