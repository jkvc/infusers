#!/usr/bin/env bash
# HTTP smoke: Klein SSE stream endpoint — progress events + final WebP.
#
# Setup: same .env as smoke.sh (needs MODAL_STREAM_URL + proxy token).
# Run:
#   ./scripts/smoke_stream.sh
#
# Writes /tmp/klein-stream.webp. JSON batch infer: ./scripts/smoke.sh
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/modal_env.sh"
_modal_env_load

URL="${MODAL_STREAM_URL:?Set MODAL_STREAM_URL in .env}"

payload='{
  "path": "klein9b.image",
  "inputs": {
    "prompt": "solid red square on white background",
    "seed": 42,
    "resolution": [512, 512]
  }
}'

echo "=== SSE stream ==="
START=$(date +%s.%N)
curl -sS -N -X POST "$URL" \
  -H "Content-Type: application/json" \
  "${MODAL_AUTH_HEADERS[@]}" \
  -d "$payload" \
  -o /tmp/klein-stream.txt \
  -w "HTTP %{http_code} time %{time_total}s\n"
END=$(date +%s.%N)
python3 - <<PY
start, end = float("$START"), float("$END")
print(f"wall: {end - start:.1f}s")
PY

python3 - <<'PY'
import base64
import json
from pathlib import Path

progress = []
result = None
for line in Path("/tmp/klein-stream.txt").read_text().splitlines():
    if not line.startswith("data:"):
        continue
    payload = json.loads(line.removeprefix("data:").strip())
    if payload["kind"] == "progress":
        progress.append(payload["progress"]["message"])
        print(f"  progress: {payload['progress']['message']}")
    elif payload["kind"] == "result":
        result = payload

if result is None:
    raise SystemExit("no result event in stream")

image_b64 = result["result"]["image"]
out = Path("/tmp/klein-stream.webp")
out.write_bytes(base64.b64decode(image_b64))
print(f"saved: {out} ({len(image_b64)} b64 chars)")
print(f"metadata: {result.get('metadata', {})}")
print(f"progress events: {len(progress)}")
PY
