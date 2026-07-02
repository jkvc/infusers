# VN SDEdit via `signal_rgba` (2026-07-01)

Latent-space signal pasteback for localized image edits. Extends the existing `klein9b.image` route — no new recipe or Modal path. Wire-format details: [`20260628-generic-modal-runner.md`](20260628-generic-modal-runner.md). Ops: [`docs/modal.md`](../docs/modal.md).

## Motivation

**Localized variation** edits a region around a click while preserving the rest of the image. Some stacks send separate signal image + mask fields; infusers folds the same semantics into one optional tensor on `FluxImageQuant`, with mask conventions (α=1 → edit freely, α=0 → preserve signal) and strict 1:1 resolution (no kontext snap, no resize).

Blend is **latent-only** during denoise — no pixel pasteback after decode.

## API

Optional kwarg on `FluxImageQuant.forward_gen` (and `ImageQuant` signature):

| Field | Shape | Semantics |
| --- | --- | --- |
| `signal_rgba` | CHW float32 [0,1], 4 channels | RGB = signal image; alpha = edit mask |

`cond_images` remains independent and can be combined with `signal_rgba`.

Modal HTTP — base64 PNG/WebP RGBA in JSON, decoded server-side:

```json
{
  "path": "klein9b.image",
  "inputs": {
    "prompt": "...",
    "seed": 43,
    "resolution": [768, 768],
    "num_steps": 20,
    "signal_rgba": "<base64 PNG RGBA>"
  },
  "translator": { "signal_rgba": "rgba_b64_to_tensor" }
}
```

`resolution` must match `signal_rgba` spatial size exactly. Typical flow: t2i at low steps → build mask client-side → vnsdedit at higher `num_steps` with the base image as signal RGB.

Route registration: `allowed_input_translators.signal_rgba → rgba_b64_to_tensor` on `klein9b.image` in `lunas_courageous_adventure.py`.

## Blend algorithm

After each Euler denoise step (`_iter_denoise`, `_iter_denoise_cached`, `_iter_denoise_cfg` in `image.py`):

```
noised_signal = signal_latent * (1 - t_prev) + noise * t_prev
pasteback_mask = (signal_mask_latent <= t_prev)
img = img * (1 - mask) + noised_signal * mask
```

Setup (`encode_signal_blend` in `vnsdedit.py`):

1. Encode signal RGB through AE (`default_prep` + float32 encode, autocast off — avoids bf16/float bias mismatch in conv).
2. Downsample mask to latent grid via bilinear interpolate.
3. Reuse the same `randn` tensor as the denoise init for noise-consistent pasteback.

Mask semantics in blend: **higher mask values → edit region** (less pasteback early in the schedule); lower values → preserve signal longer.

## Key files

| Path | Role |
| --- | --- |
| `infusers/quant/flux/vnsdedit.py` | `SignalBlendState`, validate/split/compose, blend math, `encode_signal_blend`, `compute_radial_mask` |
| `infusers/quant/flux/image.py` | `signal_rgba` param; blend hook in denoise iterators |
| `infusers/quant/api/image_base.py` | `pil_rgba_to_chw_float01`; signature on `ImageQuant` |
| `infusers/modal_app/translators/atomic.py` | `RgbaB64ToTensor` → `rgba_b64_to_tensor` |
| `tests/quant/test_vnsdedit.py` | Unit tests for blend math, validation, radial mask |
| `scripts/vnsdedit_e2e.py` | HTTP e2e: t2i (4 steps) → radial mask → vnsdedit (20 steps) × 3 samples → `tmp/vnsdedit-e2e/` |
| `infusers/modal_app/lunas_courageous_adventure.py` | Route translator allowlist; `smoke_vnsdedit` entrypoint |

`compute_radial_mask` provides localized-variation-style falloff (cosine / linear / gaussian). Used by e2e and smoke scripts on the client; not invoked inside the GPU quant path.

## Testing

```bash
uv run pytest tests/quant/test_vnsdedit.py
uv run modal deploy infusers/modal_app/lunas_courageous_adventure.py
uv run modal run infusers/modal_app/lunas_courageous_adventure.py::smoke_vnsdedit
uv run python scripts/vnsdedit_e2e.py   # needs .env proxy auth + deployed URL
```

E2e samples (768×768): watercolor street, teal drop, pine forest — radial masks at configured click points. Scratch viewer: `tmp/vnsdedit-e2e/viewer.html` (click toggles base ↔ edited).

## Deploy note

After a code-only deploy, **warm containers may still run the previous image** until they scale down (`scaledown_window=300`) or the app is stopped (`uv run modal app stop lunas-courageous-adventure -y`). If vnsdedit fails with an AE dtype error on HTTP but `modal run` smoke passes, restart containers.

## Not in scope (yet)

- External UI wiring for localized edit flows.
- `signal_rgba` on pano or other quants.
