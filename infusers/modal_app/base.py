"""Generic Modal model runner — route dispatch, translators, describe, streaming."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

import infusers.modal_app.translators.atomic as _translators_atomic  # noqa: F401 — registration
import infusers.modal_app.translators.combinators as _translators_combinators  # noqa: F401
from infusers.modal_app.log_util import (
    log_event,
    summarize_kwargs,
    summarize_output_value,
    summarize_request,
)
from infusers.modal_app.stream import (
    BoundedProgressBridge,
    encode_sse,
    run_generator_in_thread,
)
from infusers.modal_app.translators.context import TranslatorContext
from infusers.modal_app.translators.dsl import apply
from infusers.modal_app.translators.registry import TranslatorFn, apply_chain, registered_names
from infusers.quant.api.base import IntermediateEvent

DESCRIBE_PATH = "__DESCRIBE__"


@dataclass(frozen=True)
class RouteDef:
    path: str
    recipe: str
    output_key: str
    final_translators: list[TranslatorFn]
    intermediate_translators: list[TranslatorFn] = field(default_factory=list)
    allowed_input_translators: dict[str, list[str]] = field(default_factory=dict)


class RunnerError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class GenericModelRunner:
    """Platform-agnostic runner; Modal apps subclass and set ROUTES."""

    ROUTES: ClassVar[list[RouteDef]] = []

    def __init__(self) -> None:
        self._route_index: dict[str, RouteDef] = {}
        self._quants: dict[str, Any] = {}

    def init_routes(self) -> None:
        if not self.ROUTES:
            raise RuntimeError(f"{type(self).__name__}.ROUTES is empty")
        self._route_index = {route.path: route for route in self.ROUTES}
        duplicates = len(self.ROUTES) - len(self._route_index)
        if duplicates:
            raise RuntimeError("Duplicate route paths in ROUTES")

    def get_quant(self, recipe: str) -> Any:
        if recipe not in self._quants:
            from infusers import QM

            self._quants[recipe] = QM.build(recipe)
        return self._quants[recipe]

    def run(self, body: dict[str, Any]) -> dict[str, Any]:
        path = body.get("path")
        if path == DESCRIBE_PATH:
            log_event("describe_begin")
            response = self._describe()
            log_event("describe_end", routes=list(response["result"]["routes"].keys()))
            return response

        route, kwargs, ctx, wall_t0 = self._prepare_inference(body)
        quant = self.get_quant(route.recipe)
        device = ctx.device

        quant_t0 = time.perf_counter()
        log_event("quant_begin", recipe=route.recipe, device=str(device) if device else None)
        out = quant(**kwargs)
        quant_ms = int((time.perf_counter() - quant_t0) * 1000)
        log_event("quant_end", elapsed_ms=quant_ms)

        translate_t0 = time.perf_counter()
        translated = apply_chain(route.final_translators, out, ctx)
        translate_ms = int((time.perf_counter() - translate_t0) * 1000)
        log_event(
            "output_translators_applied",
            elapsed_ms=translate_ms,
            output=summarize_output_value(route.output_key, translated),
        )

        total_ms = int((time.perf_counter() - wall_t0) * 1000)
        log_event(
            "inference_end",
            path=route.path,
            recipe=route.recipe,
            elapsed_ms=total_ms,
            quant_ms=quant_ms,
            output_translate_ms=translate_ms,
        )

        return {
            "result": {route.output_key: translated},
            "metadata": self._build_metadata(
                route,
                total_ms=total_ms,
                quant_ms=quant_ms,
                output_translate_ms=translate_ms,
                device=device,
            ),
        }

    def run_stream(self, body: dict[str, Any]) -> Iterator[str]:
        route, kwargs, ctx, wall_t0 = self._prepare_inference(body)
        quant = self.get_quant(route.recipe)
        device = ctx.device

        if not hasattr(quant, "forward_gen"):
            raise RunnerError(
                f"Route {route.path!r} does not support streaming (no forward_gen)",
                status_code=400,
            )

        bridge = BoundedProgressBridge()
        thread = run_generator_in_thread(
            bridge,
            lambda: quant.forward_gen(**kwargs),
            lambda item: isinstance(item, IntermediateEvent),
        )

        quant_t0 = time.perf_counter()
        log_event("quant_stream_begin", recipe=route.recipe, device=str(device) if device else None)

        for frame in bridge.iter_frames():
            if frame.kind == "intermediate":
                translated = apply_chain(route.intermediate_translators, frame.value, ctx)
                log_event("progress_event", message=translated)
                yield encode_sse({"kind": "progress", "message": translated})
                continue

            if frame.kind == "error":
                thread.join(timeout=1)
                exc = frame.value
                if isinstance(exc, RunnerError):
                    raise exc
                raise RunnerError(str(exc), status_code=500) from exc

            quant_ms = int((time.perf_counter() - quant_t0) * 1000)
            log_event("quant_stream_end", elapsed_ms=quant_ms)

            translate_t0 = time.perf_counter()
            translated = apply_chain(route.final_translators, frame.value, ctx)
            translate_ms = int((time.perf_counter() - translate_t0) * 1000)
            log_event(
                "output_translators_applied",
                elapsed_ms=translate_ms,
                output=summarize_output_value(route.output_key, translated),
            )

            total_ms = int((time.perf_counter() - wall_t0) * 1000)
            metadata = self._build_metadata(
                route,
                total_ms=total_ms,
                quant_ms=quant_ms,
                output_translate_ms=translate_ms,
                device=device,
            )
            log_event(
                "inference_stream_end",
                path=route.path,
                recipe=route.recipe,
                elapsed_ms=total_ms,
                quant_ms=quant_ms,
                output_translate_ms=translate_ms,
            )
            yield encode_sse(
                {
                    "kind": "result",
                    "result": {route.output_key: translated},
                    "metadata": metadata,
                }
            )
            thread.join(timeout=1)
            return

        thread.join(timeout=1)

    def _prepare_inference(
        self,
        body: dict[str, Any],
    ) -> tuple[RouteDef, dict[str, Any], TranslatorContext, float]:
        path = body.get("path")
        if not isinstance(path, str) or not path:
            raise RunnerError("path is required", status_code=400)

        if path == DESCRIBE_PATH:
            raise RunnerError("Use run() for __DESCRIBE__", status_code=400)

        route = self._route_index.get(path)
        if route is None:
            raise RunnerError(f"Unknown path: {path}", status_code=404)

        raw_inputs = body.get("inputs")
        if not isinstance(raw_inputs, dict):
            raise RunnerError("inputs must be an object", status_code=400)

        translator_map = body.get("translator") or {}
        if translator_map is not None and not isinstance(translator_map, dict):
            raise RunnerError("translator must be an object when provided", status_code=400)

        wall_t0 = time.perf_counter()
        log_event("inference_begin", request=summarize_request(body), recipe=route.recipe)

        kwargs = dict(raw_inputs)
        self._require_input_translators(route, kwargs, translator_map)
        self._apply_input_translators(route, kwargs, translator_map)
        log_event("input_translators_applied", quant_kwargs=summarize_kwargs(kwargs))

        quant = self.get_quant(route.recipe)
        device = getattr(getattr(quant, "model", None), "device", None)
        ctx = TranslatorContext(device=device)
        return route, kwargs, ctx, wall_t0

    def _build_metadata(
        self,
        route: RouteDef,
        *,
        total_ms: int,
        quant_ms: int,
        output_translate_ms: int,
        device: Any,
    ) -> dict[str, Any]:
        return {
            "path": route.path,
            "recipe": route.recipe,
            "elapsed_ms": total_ms,
            "quant_ms": quant_ms,
            "output_translate_ms": output_translate_ms,
            "device": str(device) if device is not None else None,
        }

    def _require_input_translators(
        self,
        route: RouteDef,
        kwargs: dict[str, Any],
        translator_map: dict[str, Any],
    ) -> None:
        for field_name in route.allowed_input_translators:
            if field_name in kwargs and field_name not in translator_map:
                raise RunnerError(
                    f"Field {field_name!r} requires a translator entry in translator map",
                    status_code=400,
                )

    def _apply_input_translators(
        self,
        route: RouteDef,
        kwargs: dict[str, Any],
        translator_map: dict[str, Any],
    ) -> None:
        for field_name, dsl in translator_map.items():
            if field_name not in route.allowed_input_translators:
                raise RunnerError(
                    f"Field {field_name!r} does not accept input translators on this route",
                    status_code=400,
                )
            allowed = route.allowed_input_translators[field_name]
            if dsl not in allowed:
                raise RunnerError(
                    f"Translator {dsl!r} not allowed for field {field_name!r}; allowed: {allowed}",
                    status_code=400,
                )
            if field_name not in kwargs:
                raise RunnerError(
                    f"Cannot apply translator to missing field {field_name!r}",
                    status_code=400,
                )

            quant = self.get_quant(route.recipe)
            device = getattr(getattr(quant, "model", None), "device", None)
            ctx = TranslatorContext(device=device)
            kwargs[field_name] = apply(dsl, kwargs[field_name], ctx)

    def _describe(self) -> dict[str, Any]:
        routes: dict[str, Any] = {}
        for route in self.ROUTES:
            routes[route.path] = {
                "recipe": route.recipe,
                "allowed_input_translators": route.allowed_input_translators,
                "final_translators": [repr(t) for t in route.final_translators],
                "intermediate_translators": [repr(t) for t in route.intermediate_translators],
                "output_schema": {route.output_key: "string (webp base64)"},
                "stream_schema": {
                    "progress": "string message",
                    "result": {route.output_key: "string (webp base64)"},
                },
            }

        return {
            "result": {
                "endpoint_class": type(self).__name__,
                "routes": routes,
                "available_translators": registered_names(),
            },
            "metadata": {"path": DESCRIBE_PATH},
        }
