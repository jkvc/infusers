# CLAUDE.md

Guidance for AI assistants working in this repo.

## Project Overview

**infusers** — custom ML inference for jkvc: inferencer logic (MultiDiffusion, panorama, tiled modes, etc.) and private [fal.ai](https://fal.ai) serverless GPU deployments. The product is **one weight load, many parametrized algorithms** — not fixed third-party pipelines. See [`notes/`](notes/) for architecture and hosting decisions.

## Important Rules

### 1. Never Commit or Push Without Explicit Permission

You are **never allowed** to commit or push any code unless the user explicitly tells you to do so in a **separate user message**. Do not proactively commit or push changes, even if they appear complete.

### 2. Never Commit Secrets

Never commit `.env` files, API keys, or tokens. Remote fal workers use `fal secret set`; local reference values go in `.env` (gitignored).

### 3. Keep CLAUDE.md Stable

Do **not** add frequently-changing content to this file (per-app env vars, pinned model versions, deployment URLs). This file is for stable rules and conventions. Put implementation detail in code comments, README, or `notes/`.

### 4. Package Manager

This project uses **uv**. Never use bare `pip install` outside the uv-managed environment.

- Sync deps: `uv sync`
- Run a command in the venv: `uv run <cmd>`
- Add a dependency: `uv add <package>` (runtime) or `uv add --dev <package>`

### 5. fal CLI

- `fal auth login` — authenticate locally
- `fal run <app-name>` — ephemeral deployment (temporary URL, killed on Ctrl+C)
- `fal deploy <app-name>` — persistent private deployment
- `fal secret set <KEY> <value>` — secrets for remote workers
- App names are registered in `pyproject.toml` under `[tool.fal.apps]`

### 6. Repository Layout

```
infusers/
├── infusers/          # Core inferencer logic (platform-agnostic Python)
├── apps/              # fal.App deployments (one directory per endpoint)
├── tests/             # pytest
└── notes/             # Design notes (yyyymmdd-slug.md)
```

- **Heavy imports** (`torch`, `diffusers`, etc.) belong inside `setup()` or endpoint methods in fal apps, not at module level in apps that only need the fal CLI locally.
- **Pin dependency versions** in each `fal.App` `requirements` list for reproducible remote builds.
- **Share code** between fal apps via the `infusers` package or `app_files` pointing at repo paths — not ad-hoc copies.

### 7. Model Loading Convention

Weights come from Hugging Face. Use **diffusers** for standard pipeline routes; port denoise loops from BFL's [`flux2`](https://github.com/black-forest-labs/flux2) reference when building custom inferencers. Do not passthrough hosted Klein SKUs — custom algorithms are the product.

### 8. Track Tech Debt

Known shortcuts and deferred work go in [`TECH_DEBT.md`](TECH_DEBT.md). Record every conscious shortcut; delete entries once resolved.

### 9. Prose Line Wrapping

Do **not** hard-wrap paragraphs in Markdown. Write each paragraph / list item as a single unwrapped line. Blank lines still separate blocks.

### 10. Testing

Tests live in `tests/`. Run with `uv run pytest`. A pre-push hook (`.githooks/pre-push`) runs ruff, `black --check`, and pytest before every push — enable once per clone with `git config core.hooksPath .githooks`. Practice TDD for non-trivial inferencer math; skip tests for fal glue and config-only changes where they add no signal.
