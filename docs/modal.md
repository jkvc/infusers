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

After deploy, the CLI prints the web URL. Interactive API docs: `<web-url>/docs`.

HTTP smoke (cold + warm timing):

```bash
export MODAL_WEB_URL=https://<your-workspace>--lunas-courageous-adventure-<label>.modal.run
./scripts/smoke.sh
```

Example infer request:

```bash
curl -X POST "$MODAL_WEB_URL" \
  -H "Content-Type: application/json" \
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
{"kind":"progress","message":"denoise step 1/4"}
{"kind":"result","result":{"image":"<webp b64>"},"metadata":{...}}
```

Klein stream URL pattern: `<web-url>-stream.modal.run` (see deploy output).

Optional conditional images — caller must supply the input translator:

```bash
curl -X POST "$MODAL_WEB_URL" \
  -H "Content-Type: application/json" \
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
| `scripts/smoke.sh` | HTTP timing smoke test |

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
