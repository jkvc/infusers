# infusers

Custom ML inference for [jkvc](https://jkvc.ai): inferencer algorithms and private [Modal](https://modal.com) GPU deployments.

One weight load, many parametrized modes (plain t2i, MultiDiffusion, panorama, tiled inference, etc.). See [`notes/`](notes/) for architecture and hosting decisions.

## Repository layout

```
infusers/
├── infusers/
│   ├── configs/     # reqm YAML recipes (models + quants)
│   ├── model/       # Model implementations (KleinModel, …)
│   ├── quant/       # Inferencers (FluxImageQuant, …)
│   ├── scripts/     # inference_image.py CLI
│   └── modal_app/   # Modal deploy modules
├── docs/
├── scripts/         # Weight staging, upload, smoke tests
├── tests/
└── notes/
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — install: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- NVIDIA GPU + CUDA driver (for local Klein inference)
- A [Modal](https://modal.com) account (for deployments)

## Local setup

```bash
cd infusers

uv sync --dev
uv run modal setup

# Enable the pre-push hook (once per clone)
git config core.hooksPath .githooks
```

`uv sync --dev` installs flux2, torch (~3 GB of CUDA wheels), Modal CLI, and dev tools. One venv at `.venv/`.

Hugging Face auth for gated weights (`FLUX.2-dev`, `FLUX.2-klein-9B`):

1. Accept the licenses on Hugging Face
2. `uv run hf auth login` (read token; saved to `~/.cache/huggingface/token`)

## Modal deployment (Klein 9B)

Full guide: [`docs/modal.md`](docs/modal.md).

```bash
./scripts/stage_weights.sh       # one-time, ~27GB local
./scripts/upload_weights.sh      # one-time, → Modal Volume
uv run modal deploy infusers/modal_app/lunas_courageous_adventure.py
uv run modal run infusers/modal_app/lunas_courageous_adventure.py::smoke
```

## Commands

| Command | Description |
| --- | --- |
| `uv sync --dev` | Install / update all dependencies |
| `uv run hf auth login` | Hugging Face auth for gated model downloads |
| `uv run modal setup` | Authenticate Modal CLI |
| `uv run modal deploy infusers/modal_app/lunas_courageous_adventure.py` | Deploy Klein 9B |
| `uv run modal run infusers/modal_app/lunas_courageous_adventure.py::smoke` | CLI smoke (uses `modal setup`, no proxy token) |
| `./scripts/smoke.sh` | HTTP JSON smoke (needs `.env` proxy token) |
| `./scripts/smoke_stream.sh` | HTTP SSE smoke (needs `.env`) |
| `uv run ruff check .` | Lint |
| `uv run black .` | Format |
| `uv run pytest` | Unit tests |
| `uv run python -m infusers.scripts.inference_image --recipe quant/flux/klein9b/image_basic -p "…" -o out/` | Local GPU inference via reqm |

## Adding a Modal app

1. Create `infusers/modal_app/<name>.py` — thin wrapper around `QM.build(recipe)`.
2. Add quant/model YAML under `infusers/configs/`.
3. Test: `uv run modal run infusers/modal_app/<name>.py::<entrypoint>`
4. Deploy: `uv run modal deploy infusers/modal_app/<name>.py`

All call sites use `from infusers import QM` — one chokepoint, recipe name is the only variable.

## Environment

Copy `.env.example` → `.env` and fill in proxy token + lunas endpoint URLs (`.env` is gitignored).

**One Modal deploy = two URLs** (JSON + stream). Every route on that app (`klein9b.image`, future paths) shares them — the request body `"path"` picks the recipe. Adding a model is a new `RouteDef`, not a new env var. A second deploy (e.g. dummy CPU app) is optional and gets its own URL pair.

| Variable | Description |
| --- | --- |
| `MODAL_WEB_URL` | Primary app JSON endpoint (lunas) |
| `MODAL_STREAM_URL` | Primary app SSE — label `{APP_NAME}-stream` |
| `MODAL_KEY` | Proxy auth token ID (`wk-…`) |
| `MODAL_SECRET` | Proxy auth token secret (`ws-…`) |
| `MODAL_DUMMY_*` | Optional — only if HTTP-smoking `dummy_image.py` |

```bash
cp .env.example .env   # first time only
./scripts/smoke.sh     # sources .env automatically
```

Modal CLI auth (`uv run modal setup`) lives in `~/.modal.toml` (gitignored) — separate from proxy tokens. HF auth: `~/.cache/huggingface/token`.
