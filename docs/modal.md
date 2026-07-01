# Modal — Klein 9B deployment

Serverless GPU hosting for custom Klein inference. Background: [`notes/20260628-gpu-hosting-experiment.md`](../notes/20260628-gpu-hosting-experiment.md), live config: [`notes/20260628-modal-setup.md`](../notes/20260628-modal-setup.md).

## Prerequisites

- Python 3.12+, [uv](https://docs.astral.sh/uv/)
- Modal account — `uv run modal setup` (writes `~/.modal.toml`, gitignored)
- Local HF auth for staging: accept [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) and [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) licenses, then `uv run hf auth login`

Modal is a **dev dependency**. Always run the CLI as `uv run modal …` (or `alias modal='uv run modal'` in your shell).

## One-time: stage and upload weights (~27GB)

From repo root:

```bash
uv sync --dev
./scripts/stage_weights.sh          # writes weights/klein-9b/ (gitignored)
./scripts/upload_weights.sh         # → Modal Volume jkvc-klein-9b-weights
```

Upload is slow once; the Volume persists across deploys.

## Deploy and test

```bash
uv run modal deploy infusers/modal_app/lunas_courageous_adventure.py
uv run modal run infusers/modal_app/lunas_courageous_adventure.py::smoke
```

After deploy, the CLI prints the web URL (🔑 = proxy auth required). Interactive API docs: `<web-url>/docs` (also requires proxy auth headers).

## Web endpoint auth

All `web` and `web_stream` endpoints use `requires_proxy_auth=True`. Unauthenticated requests get **401** at Modal's edge — no GPU container spins up.

1. Create a [proxy auth token](https://modal.com/settings/proxy-auth-tokens) in your Modal workspace.
2. Pass `Modal-Key` (token ID) and `Modal-Secret` (token secret) on every HTTP request.

```bash
cp .env.example .env   # fill MODAL_KEY / MODAL_SECRET from proxy-auth-tokens settings
./scripts/smoke.sh
./scripts/smoke_stream.sh
```

Example infer request:

```bash
set -a && source .env && set +a
curl -X POST "$MODAL_WEB_URL" \
  -H "Content-Type: application/json" \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  -d '{
    "path": "klein9b.image",
    "inputs": {
      "prompt": "a cat",
      "seed": 42,
      "resolution": [512, 512]
    }
  }' | jq .
```

Response shape: `{ "result": { "image": "<webp base64>" }, "metadata": { ... } }`. Decode `result.image` to save a WebP file.

Streaming (SSE) — same request body, POST to the `/stream` endpoint (label varies by deploy). Events:

```json
{"kind":"progress","progress":{"message":"denoise step 1/4"}}
{"kind":"result","result":{"image":"<webp b64>"},"metadata":{...}}
```

Klein stream URL pattern: `<workspace>--lunas-courageous-adventure-stream.modal.run` (label `{APP_NAME}-stream`; see deploy output).

Optional conditional images — caller must supply the input translator:

```bash
curl -X POST "$MODAL_WEB_URL" \
  -H "Content-Type: application/json" \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  -d '{
    "path": "klein9b.image",
    "inputs": {
      "prompt": "match this style",
      "seed": 42,
      "resolution": [512, 512],
      "cond_images": ["<base64>"]
    },
    "translator": {
      "cond_images": "list_apply[imageb64_to_tensor]"
    }
  }' | jq .
```

Introspection (`__DESCRIBE__`):

```bash
curl -X POST "$MODAL_WEB_URL" \
  -H "Content-Type: application/json" \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  -d '{"path": "__DESCRIBE__"}' | jq .
```

## Local inference (same recipe)

```bash
uv run python -m infusers.scripts.inference_image \
  --recipe quant/flux/klein9b/image_basic \
  -p "solid red square on white background" \
  -o .model-out/smoke \
  --resolution 512 512 --seed 42
```

CPU-offload test recipe: `quant/flux/klein9b/image_basic_offload`.

Panorama recipe: `quant/flux/klein9b/pano_basic` — multiple slice prompts, cylindrical horizontal/vertical stitching.

```bash
uv run python -m infusers.scripts.inference_pano \
  --recipe quant/flux/klein9b/pano_basic \
  -p "warm desert dunes" -p "cool ocean horizon" \
  -o .model-out/pano \
  --resolution 512 1024 --overlap 256 --seed 42
```

Modal pano request:

```bash
curl -X POST "$MODAL_WEB_URL" \
  -H "Content-Type: application/json" \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  -d '{
    "path": "klein9b.pano",
    "inputs": {
      "prompts": ["warm desert dunes", "cool ocean horizon"],
      "seed": 42,
      "resolution": [512, 1024],
      "pano_direction": "horizontal",
      "overlap_pixels": 256
    }
  }' | jq .
```

Pano response shape: `{ "result": { "images": ["<webp base64>", ...], "direction": "...", ... }, "metadata": { ... } }`. Single-pano routes return a one-element list.

CLI pano smoke: `uv run modal run infusers/modal_app/lunas_courageous_adventure.py::smoke_pano`

## pytest (local)

Default `uv run pytest` runs unit tests only — **excludes** `gpu` and `modal` markers (fast; safe for pre-push).

| Suite | Command | When |
| --- | --- | --- |
| Unit (default) | `uv run pytest` | Every push / CI |
| Local GPU smoke | `uv run pytest -m gpu` | Machine with CUDA + staged weights |
| Modal deploy e2e | `uv run pytest -m modal` | After `uv run modal setup`; deploys dummy app |
| Deployed HTTP audit | `./scripts/smoke.sh` / `smoke_stream.sh` | Needs `.env` proxy token + live Klein URL |

For Klein/pano validation without local GPU, use Modal entrypoints (`smoke`, `smoke_pano`) or HTTP smoke scripts against the deployed app.

## Dummy runner (CPU, no weights)

Fast e2e / CI path using `quant/image_basic_dummy`:

```bash
uv run modal deploy infusers/modal_app/dummy_image.py
uv run modal run infusers/modal_app/dummy_image.py::smoke
uv run modal run infusers/modal_app/dummy_image.py::smoke_stream
```

Path: `dummy.image`. Same JSON envelope as Klein; streams progress messages then the final WebP.

## Day-to-day workflow

| Task | Command |
| --- | --- |
| Change inference logic | Edit `infusers/quant/` or model YAML, redeploy |
| Change hyperparams | Edit `infusers/configs/quant/...` YAML |
| Change weights | Re-run `stage_weights.sh` + `upload_weights.sh` (rare) |
| Validate configs | `uv run python -c "from infusers import QM; QM.validate()"` |
| Unit tests | `uv run pytest` |
| Modal e2e tests | `uv run pytest -m modal` |
| GPU smoke tests | `uv run pytest -m gpu` |
| View logs | `uv run modal app logs lunas-courageous-adventure` |
| Dashboard | https://modal.com/apps — select `lunas-courageous-adventure` |

Code-only deploys are fast (small image). Weights mount from Volume at container start.

## Key files

| Path | Role |
| --- | --- |
| `infusers/modal_app/lunas_courageous_adventure.py` | Modal app — route defs + generic runner subclass |
| `infusers/modal_app/base.py` | `GenericModelRunner` — route dispatch, describe |
| `infusers/modal_app/translators/` | Translator classes + input DSL registry |
| `infusers/model/klein.py` | `KleinModel` — flow, AE, text encoder only |
| `infusers/modal_app/dummy_image.py` | CPU dummy runner — cheap e2e |
| `infusers/modal_app/stream.py` | Bounded progress queue + SSE framing |
| `infusers/quant/api/base.py` | `TorchQuant`, `IntermediateEvent`, `FinalEvent` |
| `infusers/quant/api/image_base.py` | `ImageQuant`, `DummyImageQuant`, event types |
| `infusers/quant/flux/image.py` | `FluxImageQuant` — steps, guidance, streaming denoise loop |
| `infusers/configs/` | reqm YAML recipes |
| `infusers/__init__.py` | `QM = QuantManager(configs)` chokepoint |
| `scripts/stage_weights.sh` | Stage weights locally |
| `scripts/upload_weights.sh` | Upload to Modal Volume |
| `scripts/smoke.sh` | HTTP JSON cold/warm + describe (needs `.env`) |
| `scripts/smoke_stream.sh` | HTTP SSE stream smoke (needs `.env`) |

## Runtime behavior

- **GPU:** L40S
- **Setup:** loads `KleinModel` via reqm (~60s) — no warmup infer in setup
- **Cold first request:** setup + first infer (~70s total wall on HTTP)
- **Warm:** ~1s per request while container is up
- **`scaledown_window`:** 120s idle billing window

## Troubleshooting

**`command not found: modal`** — use `uv run modal`, not bare `modal`.

**`Missing klein weights`** — run `upload_weights.sh` (Modal) or `stage_weights.sh` (local).

**Config errors** — run `QM.validate()`; every YAML needs `# @package _global_` header.

**`Available configs: []` on Modal** — reqm YAML is not shipped by `add_local_python_source` alone; the app image must also mount `infusers/configs/` (see `add_local_dir` in `lunas_courageous_adventure.py`).
