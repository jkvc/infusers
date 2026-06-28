"""Translator registry — name → callable or arg factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from infusers.modal_app.translators.context import TranslatorContext

TranslatorFn = Callable[[Any, TranslatorContext], Any]
TranslatorFactory = Callable[[str], TranslatorFn]

_REGISTRY: dict[str, TranslatorFn | TranslatorFactory] = {}


RegisterDecorator = Callable[
    [TranslatorFn | TranslatorFactory],
    TranslatorFn | TranslatorFactory,
]


def register(name: str) -> RegisterDecorator:
    def decorator(fn: TranslatorFn | TranslatorFactory) -> TranslatorFn | TranslatorFactory:
        if name in _REGISTRY:
            raise ValueError(f"Translator already registered: {name}")
        _REGISTRY[name] = fn
        return fn

    return decorator


def lookup(name: str) -> TranslatorFn | TranslatorFactory:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown translator: {name}") from exc


def registered_names() -> list[str]:
    return sorted(_REGISTRY)


def resolve_atom(name: str, arg: str | None = None) -> TranslatorFn:
    entry = lookup(name)
    if arg is not None:
        if not callable(entry):
            raise TypeError(f"Translator {name!r} does not accept arguments")
        fn = entry(arg)
        if not callable(fn):
            raise TypeError(f"Translator factory {name!r} did not return a callable")
        return fn
    if not callable(entry):
        raise TypeError(f"Translator {name!r} requires an argument")
    result = entry("")  # type: ignore[call-arg]
    if not callable(result):
        raise TypeError(f"Translator factory {name!r} did not return a callable")
    return result


def apply_chain(steps: list[TranslatorFn], value: Any, ctx: TranslatorContext) -> Any:
    current = value
    for step in steps:
        current = step(current, ctx)
    return current
