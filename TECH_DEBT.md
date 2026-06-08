# Tech Debt

Known shortcuts and deferred work. Record every conscious shortcut here; delete entries once resolved.

## Active

### Legacy fal app removed during repo reset

**Added:** 2026-06-08

**What:** The old `apps/flux_klein_t2i` prototype (diffusers `Flux2KleinPipeline` on 9B with wrong distilled defaults) was deleted. No fal apps are registered yet.

**Why:** Repo reset to minimal scaffold before rebuilding inferencer-first architecture on fal.

**Fix:** Add new apps under `apps/` with correct Klein 4B public / 9B-KV demo split per `notes/2026-06-08-kickoff.md`.
