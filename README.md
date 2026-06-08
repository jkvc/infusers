# infusers

Custom ML inference for [jkvc](https://jkvc.ai): inferencer algorithms and private [fal.ai](https://fal.ai) GPU deployments.

One weight load, many parametrized modes (plain t2i, MultiDiffusion, panorama, tiled inference, etc.). See [`notes/`](notes/) for architecture and hosting decisions.

## Repository layout

```
infusers/
├── infusers/     # Core inferencer logic (platform-agnostic)
├── apps/         # fal.App deployments (added as endpoints ship)
├── tests/
└── notes/
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A [fal.ai](https://fal.ai) account

## Local setup

```bash
cd infusers

uv sync
uv run fal auth login

# Enable the pre-push hook (once per clone)
git config core.hooksPath .githooks
```

Copy env template and fill in values (gitignored):

```bash
cp .env.example .env
```

Store secrets for **remote fal workers** separately:

```bash
uv run fal secret set HF_TOKEN hf_xxxxxxxx   # when a gated model is deployed
```

## Commands

| Command | Description |
| --- | --- |
| `uv sync` | Install / update dependencies |
| `uv run fal auth login` | Authenticate fal CLI |
| `uv run fal run <app>` | Ephemeral deployment (dev test) |
| `uv run fal deploy <app>` | Persistent private deployment |
| `uv run ruff check .` | Lint |
| `uv run black .` | Format |
| `uv run pytest` | Unit tests |

## Adding a fal app

1. Create `apps/<name>/app.py` with a `fal.App` subclass.
2. Register in `pyproject.toml` under `[tool.fal.apps]`.
3. Test: `uv run fal run <app-name>`
4. Deploy: `uv run fal deploy <app-name>`

Shared inferencer code belongs in the `infusers` package, not duplicated across apps.

## Environment

| Variable | Where | Description |
| --- | --- | --- |
| `FAL_KEY` | Local `.env` | API key for calling private fal endpoints (`key_id:key_secret`) |
| `HF_TOKEN` | `fal secret set` + `.env` | Hugging Face read token for gated model weights on workers |

## jkvc integration

jkvc calls private fal endpoints via `@fal-ai/client` and a server proxy route. Details will live in a note once the first app ships — see [`notes/2026-06-08-kickoff.md`](notes/2026-06-08-kickoff.md).
