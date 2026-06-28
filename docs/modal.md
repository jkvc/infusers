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

Example request:

```bash
curl -X POST "$MODAL_WEB_URL" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a cat","seed":42,"width":512,"height":512}' \
  -o out.jpg
```

## Day-to-day workflow

| Task | Command |
| --- | --- |
| Change inference code | Edit `infusers/model/` or the modal app module, then redeploy |
| Change weights | Re-run `stage_weights.sh` + `upload_weights.sh` (rare) |
| View logs | `uv run modal app logs lunas-courageous-adventure` |
| Dashboard | https://modal.com/apps — select `lunas-courageous-adventure` |

Code-only deploys are fast (small image). Weights are **not** in the Docker image; they mount from the Volume at container start.

## Key files

| Path | Role |
| --- | --- |
| `infusers/modal_app/lunas_courageous_adventure.py` | Modal app: image, Volume mount, web endpoint (Klein 9B today) |
| `infusers/model/klein.py` | Klein model load + generate (flux2) |
| `scripts/stage_weights.sh` | Copy Klein safetensors + HF Qwen cache locally |
| `scripts/upload_weights.sh` | `modal volume put` to `jkvc-klein-9b-weights` |
| `scripts/smoke.sh` | POST cold + warm timing test |

## Runtime behavior

- **GPU:** L40S
- **Cold boot:** ~60–72s (load from Volume + warmup infer in `@modal.enter`)
- **Warm:** ~1s per request while container is up
- **`scaledown_window`:** 120s — GPU stays allocated (and billed) up to 2 minutes after the last request

Tuning `scaledown_window` in the modal app module trades idle cost vs likelihood of warm hits. See experiment note for cost math.

## Troubleshooting

**`command not found: modal`** — use `uv run modal`, not bare `modal`.

**`Missing klein weights at /weights/...`** — Volume empty or wrong layout; run `upload_weights.sh` and verify with `uv run modal volume ls jkvc-klein-9b-weights /klein-9b`.

**Setup fails on HF download** — staging should have populated `weights/klein-9b/hf/`; re-run `stage_weights.sh` with valid HF auth.
