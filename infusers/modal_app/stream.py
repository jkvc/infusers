"""Streaming bridge — bounded intermediate queue with non-droppable final."""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from infusers.quant.api.base import IntermediateEvent

StreamKind = Literal["intermediate", "final", "error"]


@dataclass(frozen=True)
class StreamFrame:
    kind: StreamKind
    value: Any


class BoundedProgressBridge:
    """Decouple inference (producer thread) from SSE/network (consumer).

    ``push_intermediate`` never blocks the producer: it uses ``put_nowait`` and
    drops the oldest progress event when the bounded queue is full. Slow clients
    may miss progress frames; inference and the final result are unaffected.
    """

    def __init__(self, max_intermediate: int = 8) -> None:
        self._intermediate: queue.Queue[Any] = queue.Queue(maxsize=max_intermediate)
        self._control: queue.Queue[Any] = queue.Queue()
        self._closed = False

    def push_intermediate(self, event: IntermediateEvent) -> None:
        """Enqueue progress without blocking; drop oldest on overflow."""
        try:
            self._intermediate.put_nowait(event)
        except queue.Full:
            try:
                self._intermediate.get_nowait()
                self._intermediate.put_nowait(event)
            except queue.Empty:
                pass

    def push_final(self, value: Any) -> None:
        self._control.put(("final", value))

    def push_error(self, exc: BaseException) -> None:
        self._control.put(("error", exc))

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._control.put(None)

    def iter_frames(self) -> Iterator[StreamFrame]:
        while True:
            while True:
                try:
                    yield StreamFrame("intermediate", self._intermediate.get_nowait())
                except queue.Empty:
                    break

            try:
                ctrl = self._control.get(timeout=0.05)
            except queue.Empty:
                continue

            if ctrl is None:
                while True:
                    try:
                        yield StreamFrame("intermediate", self._intermediate.get_nowait())
                    except queue.Empty:
                        return

            kind, value = ctrl
            if kind == "final":
                yield StreamFrame("final", value)
                return
            if kind == "error":
                yield StreamFrame("error", value)
                return


def run_generator_in_thread(
    bridge: BoundedProgressBridge,
    gen_factory: Callable[[], Iterator[Any]],
    is_intermediate: Callable[[Any], bool],
) -> threading.Thread:
    """Run ``forward_gen`` on a daemon thread.

    The producer drives the generator via ``for item in gen_factory()`` — it
    does not wait for the main thread to drain ``iter_frames()`` or flush SSE
    to the network. Each ``yield`` resumes immediately after ``push_intermediate``.
    """

    def producer() -> None:
        try:
            for item in gen_factory():
                if is_intermediate(item):
                    bridge.push_intermediate(item)
                else:
                    bridge.push_final(item)
                    break
            else:
                bridge.push_error(RuntimeError("generator did not yield a final result"))
        except BaseException as exc:
            bridge.push_error(exc)
        finally:
            bridge.close()

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    return thread


def encode_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
