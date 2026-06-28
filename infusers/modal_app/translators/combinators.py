"""Composable translator combinators (DSL input path)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from infusers.modal_app.translators.context import TranslatorContext
from infusers.modal_app.translators.registry import TranslatorFn, register


@register("list_apply")
def list_apply(inner: TranslatorFn):
    def _translator(value: Iterable[Any], ctx: TranslatorContext) -> list[Any]:
        return [inner(item, ctx) for item in value]

    return _translator


@register("pipe")
def pipe(*steps: TranslatorFn):
    def _translator(value: Any, ctx: TranslatorContext) -> Any:
        current = value
        for step in steps:
            current = step(current, ctx)
        return current

    return _translator
