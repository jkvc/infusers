# Modal setup (2026-06-28)

Example GPU deploy for smoke inference. Operational commands: [`docs/modal.md`](../docs/modal.md).

## Workspace

- **Modal profile:** `<workspace>` (token in `~/.modal.toml`, gitignored)
- **App name:** `lunas-courageous-adventure`
- **Dashboard:** https://modal.com/apps/<workspace>/main/deployed/lunas-courageous-adventure

## Architecture

```
Local staging (weights/klein-9b/)
        │
        ▼  modal volume put (one-time)
Modal Volume (see `VOLUME_NAME` in deploy module) → mounted at /weights
        │
        ▼  small code image (torch, flux2, infusers)
L40S container @modal.enter setup()
        │
        ▼
POST web endpoint → JSON { result, metadata } (webp base64 in result.image)
```

Wire format and route design: [`20260628-generic-modal-runner.md`](20260628-generic-modal-runner.md).

**Weight layout on Volume** (path inside container):

| Path | Contents |
| --- | --- |
| `/weights/klein-9b/klein-9b/` | `flux-2-klein-9b.safetensors`, `ae.safetensors` |
| `/weights/klein-9b/hf/` | HF hub cache for `Qwen/Qwen3-8B-FP8` |

Loader: `QM.build("quant/flux/klein9b/image_basic")` in setup with `HF_HUB_OFFLINE=1`. Modal image mounts `infusers/configs/` explicitly (see [`docs/modal.md`](../docs/modal.md)).

## Deployed endpoints

Labels: `web` = `APP_NAME`, `web_stream` = `{APP_NAME}-stream`. Requires proxy auth (`Modal-Key` / `Modal-Secret`); see [`docs/modal.md`](../docs/modal.md).

| Endpoint | URL |
| --- | --- |
| JSON | `https://<workspace>--lunas-courageous-adventure.modal.run` |
| SSE stream | `https://<workspace>--lunas-courageous-adventure-stream.modal.run` |
| Swagger | JSON URL + `/docs` |

HTTP smoke: `./scripts/smoke.sh` and `./scripts/smoke_stream.sh` (`.env` from `.env.example`).

POST JSON — see [`20260628-generic-modal-runner.md`](20260628-generic-modal-runner.md). Path: `klein9b.image`. Streaming: [`20260628-generator-quant-streaming.md`](20260628-generator-quant-streaming.md).

## Runtime settings

| Setting | Value | Rationale |
| --- | --- | --- |
| `gpu` | `L40S` | 48GB; FP8 Qwen + Klein 9B fits |
| `scaledown_window` | `300` | Stay warm 5 min after last request; balance idle cost vs burst latency |
| `timeout` | `600` | Cold setup ~60s headroom |

## Measured performance (2026-06-28)

- **Setup log:** `Quant ready` (~60s model load)
- **Cold wall (HTTP):** ~70s (setup + first infer; no warmup in setup)
- Warm wall: ~1.2s; execution ~0.8s in Modal logs
- Warm probes at +2/4/6/8 min idle: still ~1.2s (under prior 10 min window)

## Cost snapshot

L40S **$0.000542/s** (~$1.95/hr). Billed for entire container lifetime including `scaledown_window` idle.

- Cold request: ~72s → ~$0.039
- Warm infer: ~0.9s → ~$0.0005
- Idle after one request (2 min window): +120s → ~$0.065 total session

Starter plan includes **$30/mo free compute**; 27GB Volume storage is under **1 TiB free**.

## CLI

Modal is a **dev dependency**; always invoke via `uv run modal …`, not bare `modal` (unless aliased).
