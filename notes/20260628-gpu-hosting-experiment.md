# GPU hosting experiment — fal, Replicate, Modal (2026-06-28)

We needed a serverless GPU host for **custom Klein 9B inference** (BFL flux2 stack, not a hosted SKU passthrough): one weight load, future parametrized algorithms (MultiDiffusion, tiled modes, etc.), scale-to-zero, no caller-supplied HF token.

## Candidates tried

### fal.ai (original plan)

Chosen in [`2026-06-08-kickoff.md`](2026-06-08-kickoff.md): private apps, `/data` pre-seeded weights, `fal secret set` for gated models, thin `fal.App` shells over `infusers`.

We did not ship a working fal deployment in this experiment window. Gated BFL weights and `/data` staging added friction; we pivoted to Replicate as a quicker sanity-check host. fal remains a plausible long-term option if we revisit private `/data` weight layout — see kickoff note for the original architecture.

### Replicate (Cog sanity check)

**Model:** private `jkvc/klein-9b-test` on L40S.

**Bundled weights (working):** ~30GB image with Klein safetensors + Qwen3-8B-FP8 HF cache baked in.

| Run | Queued + running (wall) | `predict_time` |
| --- | --- | --- |
| Cold | ~280s (~4.7 min) | ~0.9s |
| Warm | ~28s wall (queue) / ~0.9s infer | ~0.9s |

Cold slowness was almost entirely **OCI image pull** (~30GB), not Python setup (~19s once the container was up).

**Managed weights (failed):** Cog `weights:` + `cog weights import` split code (~2.5GB push) from weights (~26GB). Push succeeded but **hosted setup never completed** — versions auto-disabled ("consistently fails to complete setup"). Root cause: `hf://` mounts flat repo files while our loader expected HF hub cache layout; we fixed the loader but Replicate's experimental managed-weights serving path still did not yield a successful run. Community models' ~1 min cold boots use `weights.replicate.delivery` (Replicate-internal CDN), not self-serve for private gated weights.

**Verdict:** Works for validation; poor fit for custom stack + cold-boot goals without Replicate-hosted CDN access. No idle GPU billing between calls (cheaper for sparse traffic).

### Modal (chosen)

**App:** `lunas-courageous-adventure` — L40S, Volume-mounted weights, custom `infusers.model.klein` flux2 stack.

| Run | Wall time | Notes |
| --- | --- | --- |
| Cold | ~72s | ~60s setup (`Klein 9B ready`), ~0.9s infer |
| Warm | ~1.2s | |

Weights live on Modal Volume `jkvc-klein-9b-weights`; image is code-only. Cold boot reads from Volume at co-located storage (~1–3 GB/s) instead of pulling a 30GB Docker image.

**Warm retention:** `scaledown_window=120` (2 min) — GPU billed while idle during that window (~$0.07 per isolated call vs ~$0.33 at 10 min).

**Cost (Starter, list L40S $0.000542/s):** ~$0.039/cold call, ~$0.0005/warm infer; ~300 scattered cold calls/month ≈ $12 GPU (within $30/mo free credit). Volume 27GB ≈ $0 (under 1 TiB free).

## Decision

**Modal** for Klein 9B inference: custom Python, Volume weights, ~4× faster cold boot than Replicate bundled, acceptable cost for ~300 calls/month. Model in `infusers/model/klein.py`, deploy in `infusers/modal_app/lunas_courageous_adventure.py`; ops in [`docs/modal.md`](../docs/modal.md).

Historical context for fal/Replicate only — not maintained deploy paths in this repo.
