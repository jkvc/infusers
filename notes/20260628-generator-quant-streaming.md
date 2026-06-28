# Generator quants and Modal streaming (2026-06-28)

Adds streaming inference to infusers: quants yield progress via `forward_gen`, the Modal runner exposes JSON + SSE endpoints, and a CPU dummy recipe enables cheap e2e without Klein weights. Wire-format runner details (translators, `__DESCRIBE__`, request shapes) live in [`20260628-generic-modal-runner.md`](20260628-generic-modal-runner.md). Ops: [`docs/modal.md`](../docs/modal.md).

## Motivation

Callers need live progress during long GPU runs (denoise steps, encode/decode phases) without blocking on the full image. The quant layer should own *what* gets yielded; the Modal runner owns *how* it crosses the wire (SSE, backpressure, translators). Blocking call sites (`quant(**kwargs)`, CLI, scripts) must keep working — they drain the same generator via a final `forward()`.

## Quant API

All generator behavior lives on `TorchQuant` in `infusers/quant/api/base.py` — no separate generator module.

| Type | Role |
| --- | --- |
| `IntermediateEvent` | frozen dataclass, `message: str` — generic progress |
| `FinalEvent` | frozen dataclass, `message: str` — base for final artifacts |
| `TorchQuant[TIntermediate, TFinal]` | abstract `forward_gen(**kwargs)`; `@final forward()` drains the stream and returns the last non-intermediate yield |
| `ImageIntermediateEvent(IntermediateEvent)` | image-domain progress (empty subclass for now — room for extra fields later) |
| `ImageOutput(FinalEvent)` | adds `image: Tensor` CHW float32 [0,1] |
| `ImageQuant` | locked image `forward_gen` signature; only override point for image quants |
| `DummyImageQuant` | trivial CPU solid-fill quant in `image_base.py` — no extra package |

**Rule:** subclasses implement `forward_gen` only. `forward()` is not overridable; it skips `IntermediateEvent` yields and returns the final artifact.

```python
# Blocking path (Modal JSON, CLI)
out: ImageOutput = quant(prompt="...", seed=42)

# Streaming path (Modal SSE) — runner calls forward_gen directly
for item in quant.forward_gen(prompt="...", seed=42):
    ...
```

## Flux Klein quant

`FluxImageQuant` (`infusers/quant/flux/image.py`) subclasses `ImageQuant`. Denoise loops are private iterators (`_iter_denoise`, `_iter_denoise_cached`, `_iter_denoise_cfg`) in the same file — instrumented copies of flux2's loops that `yield (step, total)` before each step, then `return` the denoised tensor via `StopIteration.value`.

Typical `forward_gen` sequence:

1. `encode prompt` (and `encode reference images` if `cond_images`)
2. `denoise begin (N steps)`
3. `denoise step i/N` per timestep (streamed as denoise runs, not batched after)
4. `decode image`
5. `ImageOutput(message="image ready", image=chw)`

Recipe unchanged: `quant/flux/klein9b/image_basic` (4 steps at deploy config).

## Dummy quant and recipe

`quant/image_basic_dummy` → `DummyImageQuant` in `image_base.py`. Yields `dummy: begin`, `dummy: step i/N`, then a seed-deterministic solid fill. Used by `infusers/modal_app/dummy_image.py` (CPU, no Volume, no GPU).

## Modal runner — batch vs stream

`GenericModelRunner` (`infusers/modal_app/base.py`):

| Mode | Entry | Quant call | Response |
| --- | --- | --- | --- |
| Batch | `run()` / `web` | `quant(**kwargs)` → drains `forward_gen` inside final `forward()` | single JSON `{ result, metadata }` |
| Stream | `run_stream()` / `web_stream` | `forward_gen` in a worker thread | SSE `text/event-stream` |

`RouteDef` fields (replaces old `output_translators`):

- `intermediate_translators` — e.g. `[GetAttr("message")]` on each progress event
- `final_translators` — e.g. `[GetAttr("image"), TensorToWebpB64()]`

Streaming bridge (`infusers/modal_app/stream.py`):

- `BoundedProgressBridge` — bounded intermediate queue; when full, drops **oldest** progress (never drops final)
- `run_generator_in_thread` — producer thread pushes intermediate/final into bridge
- `encode_sse` — `data: {json}\n\n` frames

SSE wire shape:

```json
{"kind":"progress","message":"denoise step 2/4"}
{"kind":"result","result":{"image":"<webp b64>"},"metadata":{...}}
```

### Structured logging

All paths emit `[runner]` JSON via `log_util.py`. Stream mode now mirrors batch on the way out:

| Event | When |
| --- | --- |
| `inference_begin` | request received |
| `input_translators_applied` | after caller input DSL |
| `quant_begin` / `quant_end` | batch infer |
| `quant_stream_begin` / `quant_stream_end` | stream infer |
| `progress_event` | each intermediate frame (stream only) |
| `output_translators_applied` | final artifact translated (batch + stream) |
| `inference_end` / `inference_stream_end` | wall-clock summary |

Base64 in logs is summarized as `{ kind, chars, bytes }` — never inlined.

## Deployed apps

| App | Path | JSON endpoint | Stream endpoint |
| --- | --- | --- | --- |
| `dummy_image.py` | `dummy.image` | `…infusers-dummy-image-dummyimagerunner-web.modal.run` | `…--stream.modal.run` (label `stream`) |
| `lunas_courageous_adventure.py` | `klein9b.image` | `…lunas-courageous-adventure-…modal.run` | `…--klein-stream.modal.run` (label `klein-stream`) |

Stream webhook labels must be unique per workspace — Klein uses `klein-stream` because dummy already took `stream`.

Klein: L40S, Volume weights, `scaledown_window=120`. Dummy: CPU-only, `scaledown_window=60`.

### Cold vs warm (Klein, Jun 2026)

Measured after redeploy with 512×512, 4-step recipe:

| Phase | Time |
| --- | --- |
| `@modal.enter()` — `QM.build` + weight load | ~19–30s (`Runner ready …`) |
| First infer (cold container) | ~6–7s execution |
| First HTTP wall | ~32–45s |
| Warm infer | ~0.7–0.85s |
| Warm HTTP | ~0.9–1.1s |

Older notes cited ~70s cold; current numbers are roughly half that (cached model build, faster image).

## Tests

| Suite | Coverage |
| --- | --- |
| `tests/quant/test_generator.py` | event types, `forward` drain, `DummyImageQuant`, reqm build |
| `tests/modal_app/test_stream.py` | bridge drop policy, SSE encoding, thread bridge |
| `tests/modal_app/test_runner.py` | batch + stream dispatch, `progress_event` logging |
| `tests/modal_app/test_modal_e2e.py` | `modal deploy` + `modal run` smokes on dummy (`pytest -m modal`) |

HTTP validation: `curl -X POST` to both `web` and `web_stream` URLs (e2e tests use `run_remote` / `run_stream_remote` — same runner code, different entrypoint).

## File map (new/changed)

```
infusers/quant/api/
├── base.py           TorchQuant + IntermediateEvent + FinalEvent + final forward()
└── image_base.py     ImageQuant, DummyImageQuant, ImageOutput, ImageIntermediateEvent

infusers/quant/flux/image.py     FluxImageQuant.forward_gen + denoise iterators

infusers/modal_app/
├── base.py           run(), run_stream(), RouteDef translators
├── stream.py         bounded queue + SSE
├── dummy_image.py    CPU Modal app
└── lunas_courageous_adventure.py   Klein + web_stream (klein-stream label)

infusers/configs/quant/image_basic_dummy.yaml
```

Removed during consolidation: `quant/api/generator.py`, `quant/api/image_generator.py`, `quant/flux/denoise_loop.py`, `quant/dummy/image.py`.
