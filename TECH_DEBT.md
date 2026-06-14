# Tech Debt

Known shortcuts and deferred work. Record every conscious shortcut here; delete entries once resolved.

## Active

### Legacy fal app removed during repo reset

**Added:** 2026-06-08

**What:** The old `apps/flux_klein_t2i` prototype (diffusers `Flux2KleinPipeline` on 9B with wrong distilled defaults) was deleted. No fal apps are registered yet.

**Why:** Repo reset to minimal scaffold before rebuilding inferencer-first architecture on fal.

**Fix:** Add new apps under `apps/` with correct Klein 4B public / 9B-KV demo split per `notes/2026-06-08-kickoff.md`.

### Ampere GPUs use bf16 Qwen3 text encoder in vanilla script

**Added:** 2026-06-13

**What:** `scripts/vanilla_inference_klein.py` loads `Qwen/Qwen3-{4,8}B` in bf16 on GPUs with compute capability < 8.9 (e.g. RTX 3090). BFL defaults to `Qwen3-*-FP8`, which requires Ada (4090+) or newer.

**Why:** FP8 kernels are unavailable on Ampere; without the fallback the script fails at load time.

**Fix:** Port fallback into shared `infusers` loader once inferencer package exists; revisit if BFL ships an Ampere-compatible text encoder path.
