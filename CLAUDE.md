# CLAUDE.md

## Important Rules

### 1. Never Commit or Push Without Explicit Permission
You are **never allowed** to commit or push any code unless the user explicitly tells you to do so in a **separate user message**.

### 2. Never Commit Secrets
Never commit `.env` files, API keys, or tokens. All secrets are managed via `fal secret set` for remote workers and `.env` (gitignored) for local reference.

### 3. Keep CLAUDE.md Stable
Do **not** add frequently-changing content to this file. Use code comments or README for implementation details.

### 4. Language & Runtime
This is a **Python** project. No npm/yarn/pnpm. Local development uses `pip` with a virtual environment. Remote execution uses fal's serverless GPU workers.

### 5. fal CLI Commands
- `fal run <app-name>` -- ephemeral deployment for testing (temporary URL, killed on Ctrl+C)
- `fal deploy <app-name>` -- persistent production deployment
- `fal secret set <KEY> <value>` -- store secrets for remote workers
- App names are defined in `pyproject.toml` under `[tool.fal.apps]`

### 6. App Structure Convention
Each app lives in its own directory under `apps/`. Shared utilities go in `common/`. Each app file defines a single `fal.App` subclass. Heavy imports (`torch`, `diffusers`, etc.) go inside `setup()` or endpoint methods, not at module level.

### 7. Dependency Pinning
Always pin dependency versions in the `requirements` list of each `fal.App` class for reproducible remote builds.
