# Modal setup — Klein 9B (2026-06-28)

Current production path for Klein 9B smoke inference. Operational commands: [`docs/modal.md`](../docs/modal.md).

## Workspace

- **Modal profile:** `kevinehc` (token in `~/.modal.toml`, gitignored)
- **App name:** `jkvc-klein-9b`
- **Dashboard:** https://modal.com/apps/kevinehc/main/deployed/jkvc-klein-9b

## Architecture

```
Local staging (weights/klein-9b/)
        │
        ▼  modal volume put (one-time)
Modal Volume: jkvc-klein-9b-weights  →  mounted at /weights
        │
        ▼  small code image (torch, flux2, infusers)
L40S container @modal.enter setup()
        │
        ▼
POST web endpoint → JPEG
```

**Weight layout on Volume** (path inside container):

| Path | Contents |
| --- | --- |
| `/weights/klein-9b/klein-9b/` | `flux-2-klein-9b.safetensors`, `ae.safetensors` |
| `/weights/klein-9b/hf/` | HF hub cache for `Qwen/Qwen3-8B-FP8` |

Loader: `infusers.klein.load_pipeline` with `HF_HUB_OFFLINE=1`.

## Deployed endpoints

- **Web API:** `https://kevinehc--jkvc-klein-9b-klein9b-web.modal.run`
- **Swagger:** same URL + `/docs`

POST JSON: `{"prompt": "...", "seed": 42, "width": 512, "height": 512}` → JPEG bytes.

## Runtime settings

| Setting | Value | Rationale |
| --- | --- | --- |
| `gpu` | `L40S` | 48GB; FP8 Qwen + Klein 9B fits |
| `scaledown_window` | `120` | Stay warm 2 min after last request; balance idle cost vs burst latency |
| `timeout` | `600` | Cold setup ~60s headroom |

## Measured performance (2026-06-28)

- Setup log: `Klein 9B ready in ~60s`
- Cold wall (HTTP): ~72s
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
