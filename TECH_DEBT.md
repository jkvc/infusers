# Tech Debt

Known shortcuts and deferred work. Record every conscious shortcut here; delete entries once resolved.

## Active

### Klein 4B public endpoint not yet deployed

**Added:** 2026-06-08 (updated 2026-06-28)

**What:** Only Klein 9B smoke inference is deployed on Modal (`infusers/modal_app/lunas_courageous_adventure.py`). Klein 4B public / 9B-KV demo split from kickoff note is not built yet.

**Why:** Hosting experiment prioritized 9B cold-boot validation before multi-endpoint rollout.

**Fix:** Add another `infusers/modal_app/*.py` per `notes/2026-06-08-kickoff.md` once inferencer modes land.

### Ampere GPUs use bf16 Qwen3 text encoder in vanilla script

**Added:** 2026-06-13

**What:** `scripts/vanilla_inference_klein.py` loads `Qwen/Qwen3-{4,8}B` in bf16 on GPUs with compute capability < 8.9 (e.g. RTX 3090). BFL defaults to `Qwen3-*-FP8`, which requires Ada (4090+) or newer.

**Why:** FP8 kernels are unavailable on Ampere; without the fallback the script fails at load time.

**Fix:** Port fallback into shared `infusers` loader once inferencer package exists; revisit if BFL ships an Ampere-compatible text encoder path.
