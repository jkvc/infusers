# infusers

Custom ML inference — load a model once, run many algorithms.

infusers separates **models** (weights and modules), **quants** (parametrized inference algorithms), and **deployment** (local GPU, Modal, or your own host). Recipes are YAML configs managed by [reqm](https://pypi.org/project/reqm/); all entry points use `QM.build("…")`.

MIT licensed. See [`docs/modal.md`](docs/modal.md) for optional serverless GPU deployment.

## Highlights

- **Shared model, many modes** — reqm caches model instances so multiple quants reuse one weight load
- **Pluggable algorithms** — text-to-image, panorama stitching, localized latent edit, streaming progress, and more via the quant layer
- **Flexible wire format** — generic Modal runner with route paths, declarative input translators, JSON and SSE responses
- **Deploy anywhere** — local CLI, example Modal app, or embed the Python package in your own service

## Capabilities

| Capability | Status |
| --- | --- |
| Text-to-image | ✅ |
| Panorama stitching | ✅ |
| Localized edit (`signal_rgba` / vnsdedit) | ✅ |
| MultiDiffusion, general tiled inference | Planned |

Architecture and design notes: [`notes/`](notes/).

## Repository layout

```
infusers/
├── infusers/
│   ├── configs/     # reqm YAML recipes (models + quants)
│   ├── model/       # Model implementations (weights + modules)
│   ├── quant/       # Quants (FluxImageQuant, FluxPanoramaQuant, …)
│   ├── scripts/     # inference_image.py, inference_pano.py CLIs
│   └── modal_app/   # Modal deploy modules
├── docs/
├── scripts/         # Weight staging, upload, smoke tests
├── tests/
└── notes/
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — install: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- NVIDIA GPU + CUDA driver (for local inference with staged weights)
- A [Modal](https://modal.com) account (optional, for serverless GPU deployment)

## Local setup

```bash
cd infusers

uv sync --dev
uv run modal setup   # optional — only if deploying to Modal

# Enable the pre-push hook (once per clone)
git config core.hooksPath .githooks
```

`uv sync --dev` installs torch (~3 GB of CUDA wheels), Modal CLI, and dev tools. One venv at `.venv/`.

### Weights

This repo does not ship model weights. Accept the Hugging Face licenses for the models referenced by your recipes, then:

1. `uv run hf auth login` (read token; saved to `~/.cache/huggingface/token`)
2. Stage locally: `./scripts/stage_weights.sh` (see [`docs/modal.md`](docs/modal.md))
3. Or upload to a Modal Volume for serverless deploy: `./scripts/upload_weights.sh`

## Modal deployment (optional)

Full guide: [`docs/modal.md`](docs/modal.md).

```bash
./scripts/stage_weights.sh       # one-time, ~27GB local
./scripts/upload_weights.sh      # one-time, → Modal Volume
uv run modal deploy infusers/modal_app/lunas_courageous_adventure.py
uv run modal run infusers/modal_app/lunas_courageous_adventure.py::smoke
```

The example app name (`lunas-courageous-adventure`) and Volume name are constants in the deploy module — rename them for your own workspace.

## Commands

| Command | Description |
| --- | --- |
| `uv sync --dev` | Install / update all dependencies |
| `uv run hf auth login` | Hugging Face auth for gated model downloads |
| `uv run modal setup` | Authenticate Modal CLI |
| `uv run modal deploy infusers/modal_app/lunas_courageous_adventure.py` | Deploy example GPU app |
| `uv run modal run infusers/modal_app/lunas_courageous_adventure.py::smoke` | CLI smoke (uses `modal setup`, no proxy token) |
| `uv run modal run infusers/modal_app/lunas_courageous_adventure.py::smoke_pano` | CLI pano smoke |
| `./scripts/smoke.sh` | HTTP JSON smoke (needs `.env` proxy token) |
| `./scripts/smoke_stream.sh` | HTTP SSE smoke (needs `.env`) |
| `uv run ruff check .` | Lint |
| `uv run black .` | Format |
| `uv run pytest` | Unit tests (excludes `-m gpu` and `-m modal`) |
| `uv run pytest -m gpu` | Local GPU smoke (needs CUDA + staged weights) |
| `uv run pytest -m modal` | Modal deploy + run e2e (needs `uv run modal setup`) |
| `uv run python -m infusers.scripts.inference_image --recipe quant/flux/klein9b/image_basic -p "…" -o out/` | Local GPU t2i via reqm |
| `uv run python -m infusers.scripts.inference_pano --recipe quant/flux/klein9b/pano_basic -p "slice A" -p "slice B" -o out/` | Local GPU panorama via reqm |

## Adding a Modal app

1. Create `infusers/modal_app/<name>.py` — thin wrapper around `QM.build(recipe)`.
2. Add quant/model YAML under `infusers/configs/`.
3. Test: `uv run modal run infusers/modal_app/<name>.py::<entrypoint>`
4. Deploy: `uv run modal deploy infusers/modal_app/<name>.py`

All call sites use `from infusers import QM` — one chokepoint, recipe name is the only variable.

## Environment

Copy `.env.example` → `.env` and fill in proxy token + deployed app URLs (`.env` is gitignored).

**One Modal deploy = two URLs** (JSON + stream). Every route on that app shares them — the request body `"path"` picks the recipe. Adding a route is a new `RouteDef`, not a new env var. A second deploy (e.g. dummy CPU app) is optional and gets its own URL pair.

| Variable | Description |
| --- | --- |
| `MODAL_WEB_URL` | Primary app JSON endpoint |
| `MODAL_STREAM_URL` | Primary app SSE — label `{APP_NAME}-stream` |
| `MODAL_KEY` | Proxy auth token ID (`wk-…`) |
| `MODAL_SECRET` | Proxy auth token secret (`ws-…`) |
| `MODAL_DUMMY_*` | Optional — only if HTTP-smoking `dummy_image.py` |

```bash
cp .env.example .env   # first time only
./scripts/smoke.sh     # sources .env automatically
```

Modal CLI auth (`uv run modal setup`) lives in `~/.modal.toml` (gitignored) — separate from proxy tokens. HF auth: `~/.cache/huggingface/token`.
