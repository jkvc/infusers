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

| Variable | Where | Description |
| --- | --- | --- |
| `MODAL_WEB_URL` | shell | Deployed web endpoint for `./scripts/smoke.sh` |

Auth: `uv run hf auth login` (Hugging Face) and `uv run modal setup` (Modal) — credentials in `~/.cache/huggingface/token` and `~/.modal.toml` (gitignored).

## jkvc integration

jkvc will call the private Modal web endpoint (or a future proxy route). See [`notes/20260628-modal-setup.md`](notes/20260628-modal-setup.md) for the current endpoint and [`notes/2026-06-08-kickoff.md`](notes/2026-06-08-kickoff.md) for original architecture context.
