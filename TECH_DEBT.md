# Tech Debt

Known shortcuts and deferred work. Record every conscious shortcut here; delete entries once resolved.

## Active

### GPU memory snapshots not viable for Klein 9B on Modal

**Added:** 2026-06-30

**What:** Modal GPU memory snapshots tested on L40S/H100/A100 with Klein 9B. L40S/H100: snapshot create fails after load. A100: create works, restore fails when HF cache on Volume (xet logs on 9p). Cold boot ~60s is Volume read I/O (~27GB); parallel load saves ~4s (now default in `KleinModel`); local copy and CPU-only snapshots did not help.

**Why:** Alpha GPU snapshot feature; Klein GPU state too large/incompatible on most GPUs; Volume mount is the cold-start bottleneck.

**Fix:** No snapshots on `lunas_courageous_adventure`. Use `scaledown_window=300`, `parallel_load=True`, or accept ~60s cold.

### Klein 4B public endpoint not yet deployed

**Added:** 2026-06-08 (updated 2026-06-28)

**What:** Only Klein 9B smoke inference is deployed on Modal (`infusers/modal_app/lunas_courageous_adventure.py`). Klein 4B public / 9B-KV demo split from kickoff note is not built yet.

**Why:** Hosting experiment prioritized 9B cold-boot validation before multi-endpoint rollout.

**Fix:** Add another `infusers/modal_app/*.py` per `notes/2026-06-08-kickoff.md` once additional quant modes land.
