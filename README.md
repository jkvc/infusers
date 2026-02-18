# infusers

Custom ML inference services deployed on [fal.ai](https://fal.ai). Each app under `apps/` is an independently deployable serverless GPU endpoint.

## Repository Structure

```
infusers/
├── CLAUDE.md                        # Agent rules
├── README.md
├── .gitignore
├── pyproject.toml                   # Project config + [tool.fal.apps] registry
├── common/                          # Shared utilities across apps
│   └── __init__.py
└── apps/
    └── flux_klein_t2i/              # FLUX.2 Klein 9B text-to-image
        └── app.py
```

## Prerequisites

- Python 3.11+
- A [fal.ai](https://fal.ai) account
- A [HuggingFace](https://huggingface.co) account with access to gated models

### Accept the FLUX.2 Klein 9B License

The model is gated. You must accept the license before deploying:

1. Go to [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
2. Click "Agree and access repository"
3. Generate a HuggingFace token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with `read` permission

## Local Setup

```bash
# Clone and enter the repo
cd infusers

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the project (installs fal CLI)
pip install -e .

# Authenticate with fal
fal auth login

# Store your HuggingFace token as a fal secret (used by remote workers)
fal secret set HF_TOKEN hf_xxxxxxxxxxxxxxxxxx
```

Optionally, create a `.env` file for local reference (gitignored):

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxx
```

## Development

Run an app as an ephemeral deployment (temporary URL, killed on Ctrl+C):

```bash
# Using the app name from pyproject.toml
fal run flux-klein-t2i

# Or using the direct file path
fal run apps/flux_klein_t2i/app.py::FluxKleinT2I
```

This spins up a remote GPU worker, loads the model, and gives you a temporary URL with an auto-generated playground UI where you can test prompts. The worker is destroyed when you stop the command.

## Production Deployment

Deploy a persistent endpoint:

```bash
fal deploy flux-klein-t2i
```

This creates a permanent endpoint at `https://fal.run/<your-username>/flux-klein-t2i`. Redeploy the same command to update.

All deployments are **private** by default -- only authenticated requests with your `FAL_KEY` can call them.

### Deployment Options

```bash
# Deploy with rolling strategy (zero-downtime updates)
fal deploy flux-klein-t2i --strategy rolling
```

### Calling a Private Endpoint

Private endpoints require a `FAL_KEY` for authentication. Get one from [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys).

**curl:**

```bash
curl -X POST "https://fal.run/<your-username>/flux-klein-t2i" \
  -H "Authorization: Key $FAL_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat holding a sign that says hello world"}'
```

**Python (fal client):**

```python
import fal

# Uses FAL_KEY env var automatically
result = fal.run("<your-username>/flux-klein-t2i", arguments={
    "prompt": "A cat holding a sign that says hello world",
})
print(result["images"][0]["url"])
```

**JavaScript (@fal-ai/client):**

```typescript
import { fal } from "@fal-ai/client";

// Set FAL_KEY env var, or configure explicitly:
// fal.config({ credentials: "your_key_id:your_key_secret" });

const result = await fal.subscribe("<your-username>/flux-klein-t2i", {
  input: { prompt: "A cat holding a sign that says hello world" },
});
console.log(result.data.images[0].url);
```

### Managing Deployments

```bash
# List all deployed apps
fal apps list

# Delete a deployment
fal apps delete <app-id>
```

## App Configuration

Key settings in `apps/flux_klein_t2i/app.py`:

| Setting | Value | Meaning |
|---|---|---|
| `machine_type` | `GPU-A100` | 40GB VRAM, fits the ~20GB model comfortably |
| `keep_alive` | `300` | Worker stays warm for 5 min after last request |
| `min_concurrency` | `0` | Scale to zero when idle (no cost) |
| `max_concurrency` | `1` | One worker max (personal demo) |

### Cost

- **Idle**: $0 (scales to zero)
- **Cold start**: ~30-60s on first request (model download + load)
- **Warm request**: ~2-5s for generation
- **keep_alive cost**: ~$0.09 per 5-min warm window (A100 rate)

## Adding a New App

1. Create a new directory under `apps/`:
   ```
   apps/my_new_app/
   └── app.py
   ```

2. Define a `fal.App` subclass in `app.py`:
   ```python
   import fal
   from pydantic import BaseModel, Field

   class Input(BaseModel):
       prompt: str = Field(description="...")

   class Output(BaseModel):
       result: str = Field(description="...")

   class MyNewApp(fal.App):
       machine_type = "GPU-A100"
       min_concurrency = 0
       max_concurrency = 1
       requirements = ["torch==2.6.0", ...]

       def setup(self):
           import torch
           # Load model...

       @fal.endpoint("/")
       async def run(self, input: Input) -> Output:
           # Inference logic...
           return Output(result="...")
   ```

3. Register it in `pyproject.toml`:
   ```toml
   [tool.fal.apps]
   flux-klein-t2i = { ref = "apps/flux_klein_t2i/app.py::FluxKleinT2I", auth = "private" }
   my-new-app = { ref = "apps/my_new_app/app.py::MyNewApp", auth = "private" }
   ```

4. Test: `fal run my-new-app`
5. Deploy: `fal deploy my-new-app`

### Sharing Code Between Apps

Put shared utilities in `common/`. Reference them from your app using `app_files`:

```python
class MyApp(fal.App):
    app_files = ["../../common"]
    app_files_context_dir = "../../"
    # ...

    @fal.endpoint("/")
    async def run(self, input: Input) -> Output:
        from common.sampler import my_custom_sampler
        # ...
```

## Connecting to the Next.js Frontend (jkvc repo)

To call these fal endpoints from the Vercel-hosted Next.js site:

### 1. Install dependencies in jkvc

```bash
cd ../jkvc
pnpm add @fal-ai/client @fal-ai/server-proxy
```

### 2. Create the proxy route

Create `app/api/fal/proxy/route.ts`:

```typescript
import { route } from "@fal-ai/server-proxy/nextjs";

export const { GET, POST, PUT } = route;
```

This proxies fal requests through your server so the `FAL_KEY` stays secret.

### 3. Set environment variables

Add `FAL_KEY` to your Vercel project:

1. Go to [fal.ai dashboard](https://fal.ai/dashboard) > Keys
2. Create an API key
3. In Vercel dashboard > Settings > Environment Variables, add `FAL_KEY`
4. For local dev, add it to `jkvc/.env.local`: `FAL_KEY=your_key_id:your_key_secret`

### 4. Call the endpoint from frontend code

```typescript
import { fal } from "@fal-ai/client";

fal.config({ proxyUrl: "/api/fal/proxy" });

const result = await fal.subscribe("<your-username>/flux-klein-t2i", {
  input: {
    prompt: "A cat in a field of sunflowers",
    num_inference_steps: 50,
    guidance_scale: 4.0,
  },
});

console.log(result.data.images[0].url);
```

## Secrets Reference

| Secret | Where | How |
|---|---|---|
| `HF_TOKEN` | fal remote workers | `fal secret set HF_TOKEN <value>` |
| `FAL_KEY` | Local / CI for calling private endpoints | `export FAL_KEY=your_key_id:your_key_secret` |
| `FAL_KEY` | jkvc Vercel project (frontend proxy) | Vercel dashboard env vars |
