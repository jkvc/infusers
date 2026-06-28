# Tech Debt

Known shortcuts and deferred work. Record every conscious shortcut here; delete entries once resolved.

## Active

### Klein 4B public endpoint not yet deployed

**Added:** 2026-06-08 (updated 2026-06-28)

**What:** Only Klein 9B smoke inference is deployed on Modal (`infusers/modal_app/lunas_courageous_adventure.py`). Klein 4B public / 9B-KV demo split from kickoff note is not built yet.

**Why:** Hosting experiment prioritized 9B cold-boot validation before multi-endpoint rollout.

**Fix:** Add another `infusers/modal_app/*.py` per `notes/2026-06-08-kickoff.md` once inferencer modes land.
