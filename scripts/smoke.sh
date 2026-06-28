#!/usr/bin/env bash
# Cold + warm timing smoke test against deployed Modal web endpoint.
#
# Usage:
#   MODAL_WEB_URL=https://<workspace>--lunas-courageous-adventure-<label>.modal.run ./scripts/smoke.sh
set -euo pipefail

URL="${MODAL_WEB_URL:?Set MODAL_WEB_URL to your @modal.fastapi_endpoint URL}"

payload='{"prompt":"solid red square on white background","seed":42,"resolution":[512,512]}'

echo "=== cold-ish request ==="
START=$(date +%s.%N)
curl -sS -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "$payload" \
  -o /tmp/klein-cold.jpg \
  -w "HTTP %{http_code} size %{size_download} time %{time_total}s\n"
END=$(date +%s.%N)
python3 - <<PY
import os
start, end = float("$START"), float("$END")
print(f"wall: {end - start:.1f}s")
print(f"saved: /tmp/klein-cold.jpg ({os.path.getsize('/tmp/klein-cold.jpg')} bytes)")
PY

sleep 2

echo "=== warm request ==="
START=$(date +%s.%N)
curl -sS -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "$payload" \
  -o /tmp/klein-warm.jpg \
  -w "HTTP %{http_code} size %{size_download} time %{time_total}s\n"
END=$(date +%s.%N)
python3 - <<PY
start, end = float("$START"), float("$END")
print(f"wall: {end - start:.1f}s")
PY
