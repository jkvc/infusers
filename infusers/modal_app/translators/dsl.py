"""Bracket DSL parser for translator expressions."""

from __future__ import annotations

import re
from typing import Any

from infusers.modal_app.translators.context import TranslatorContext
from infusers.modal_app.translators.registry import TranslatorFn, lookup, resolve_atom

_IDENT = r"[a-zA-Z_][a-zA-Z0-9_]*"
_QUOTED = r"'([^']*)'|\"([^\"]*)\""


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def _parse_atom(text: str) -> TranslatorFn:
    text = text.strip()
    match = re.fullmatch(rf"({_IDENT})(?:\((.+)\))?", text)
    if not match:
        raise ValueError(f"Invalid translator atom: {text!r}")
    name = match.group(1)
    arg: str | None = None
    if match.group(2) is not None:
        arg_match = re.fullmatch(_QUOTED, match.group(2).strip())
        if not arg_match:
            raise ValueError(f"Invalid translator argument: {match.group(2)!r}")
        arg = arg_match.group(1) or arg_match.group(2)
    return resolve_atom(name, arg)


def _parse_combinator(text: str) -> TranslatorFn:
    text = text.strip()
    bracket = re.fullmatch(rf"({_IDENT})\[(.*)\]", text, flags=re.DOTALL)
    if not bracket:
        return _parse_atom(text)

    name = bracket.group(1)
    inner_text = bracket.group(2).strip()
    entry = lookup(name)

    if name == "list_apply":
        inner = parse(inner_text)
        result = entry(inner)  # type: ignore[operator]
        if not callable(result):
            raise TypeError("list_apply factory did not return a callable")
        return result

    if name == "pipe":
        inners = [_parse_combinator(part) for part in _split_top_level_commas(inner_text)]
        result = entry(*inners)  # type: ignore[operator]
        if not callable(result):
            raise TypeError("pipe factory did not return a callable")
        return result

    raise ValueError(f"Unsupported combinator: {name!r}")


def parse(text: str) -> TranslatorFn:
    """Parse a bracket DSL string into an executable translator."""
    return _parse_combinator(text.strip())


def apply(text: str, value: Any, ctx: TranslatorContext) -> Any:
    return parse(text)(value, ctx)
