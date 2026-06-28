# CLAUDE.md

Guidance for AI assistants working in this repo.

## Project Overview

**infusers** — custom ML inference for jkvc: inferencer logic (MultiDiffusion, panorama, tiled modes, etc.) and private [Modal](https://modal.com) GPU deployments. The product is **one weight load, many parametrized algorithms** — not fixed third-party pipelines. See [`notes/`](notes/) for architecture and hosting decisions. Modal ops: [`docs/modal.md`](docs/modal.md).

## Important Rules

### 1. Never Commit or Push Without Explicit Permission

You are **never allowed** to commit or push any code unless the user explicitly tells you to do so in a **separate user message**. Do not proactively commit or push changes, even if they appear complete.

### 2. Never Commit Secrets

Never commit `.env` files, API keys, or tokens. Modal auth lives in `~/.modal.toml`; Hugging Face auth via `uv run hf auth login` (`~/.cache/huggingface/token`). Both are gitignored.

### 3. Keep CLAUDE.md Stable

Do **not** add frequently-changing content to this file (per-app env vars, pinned model versions, deployment URLs). This file is for stable rules and conventions. Put implementation detail in code comments, README, or `notes/`.

### 4. Package Manager

This project uses **uv**. Never use bare `pip install` outside the uv-managed environment.

- Sync deps: `uv sync`
- Run a command in the venv: `uv run <cmd>`
- Add a dependency: `uv add <package>` (runtime) or `uv add --dev <package>`

### 5. Modal CLI

Modal is a **dev dependency**. Always run `uv run modal …`, not bare `modal`.

- `uv run modal setup` — one-time local auth
- `uv run modal deploy infusers/modal_app/<name>.py` — deploy app
- `uv run modal run infusers/modal_app/<name>.py::<fn>` — run a local entrypoint against deployed infra

See [`docs/modal.md`](docs/modal.md) for staging weights, Volume upload, and smoke tests.

### 6. Repository Layout

```
infusers/
├── infusers/
│   ├── configs/       # reqm YAML (models + quants)
│   ├── model/         # Model implementations (KleinModel, …)
│   ├── quant/         # Inferencers (FluxImageQuant, …)
│   ├── scripts/       # inference_image.py CLI
│   └── modal_app/     # Modal deploy modules
├── docs/              # Operational guides (e.g. modal.md)
├── scripts/           # Weight staging, upload, smoke tests
├── tests/             # pytest
└── notes/             # Design notes (yyyymmdd-slug.md)
```

- **reqm chokepoint:** `from infusers import QM` then `QM.build("<recipe>")` — all call sites (Modal, CLI, tests).
- **Heavy imports** (`torch`, etc.) belong inside quant/model code and `@modal.enter()`, not at module level in deploy modules where avoidable.
- **Pin dependency versions** in Modal image `pip_install` lists for reproducible remote builds.
- **YAML configs** must start with `# @package _global_`; run `QM.validate()` in tests/CI.

### 7. Model Loading Convention

Weights come from Hugging Face. Use **diffusers** for standard pipeline routes; port denoise loops from BFL's [`flux2`](https://github.com/black-forest-labs/flux2) reference when building custom inferencers. Do not passthrough hosted Klein SKUs — custom algorithms are the product.

### 8. Track Tech Debt

Known shortcuts and deferred work go in [`TECH_DEBT.md`](TECH_DEBT.md). Record every conscious shortcut; delete entries once resolved.

### 9. Prose Line Wrapping

Do **not** hard-wrap paragraphs in Markdown. Write each paragraph / list item as a single unwrapped line. Blank lines still separate blocks.

### 10. Testing

Tests live in `tests/`. Run with `uv run pytest`. A pre-push hook (`.githooks/pre-push`) runs ruff, `black --check`, and pytest before every push — enable once per clone with `git config core.hooksPath .githooks`. Practice TDD for non-trivial inferencer math; skip tests for Modal glue and config-only changes where they add no signal.
