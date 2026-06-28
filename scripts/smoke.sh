#!/usr/bin/env bash
# Cold + warm timing smoke test against deployed Modal web endpoint.
#
# Usage:
#   MODAL_WEB_URL=https://<workspace>--lunas-courageous-adventure-<label>.modal.run ./scripts/smoke.sh
set -euo pipefail

URL="${MODAL_WEB_URL:?Set MODAL_WEB_URL to your @modal.fastapi_endpoint URL}"

payload='{
  "path": "klein9b.image",
  "inputs": {
    "prompt": "solid red square on white background",
    "seed": 42,
    "resolution": [512, 512]
  }
}'

decode_webp() {
  local json_file="$1"
  local out_file="$2"
  python3 - <<PY
import base64, json, sys
data = json.load(open("$json_file"))
b64 = data["result"]["image"]
open("$out_file", "wb").write(base64.b64decode(b64))
print(f"saved: $out_file ({len(b64)} b64 chars)")
print(f"metadata: {data.get('metadata', {})}")
PY
}

echo "=== cold-ish request ==="
START=$(date +%s.%N)
curl -sS -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "$payload" \
  -o /tmp/klein-cold.json \
  -w "HTTP %{http_code} size %{size_download} time %{time_total}s\n"
END=$(date +%s.%N)
python3 - <<PY
start, end = float("$START"), float("$END")
print(f"wall: {end - start:.1f}s")
PY
decode_webp /tmp/klein-cold.json /tmp/klein-cold.webp

sleep 2

echo "=== warm request ==="
START=$(date +%s.%N)
curl -sS -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "$payload" \
  -o /tmp/klein-warm.json \
  -w "HTTP %{http_code} size %{size_download} time %{time_total}s\n"
END=$(date +%s.%N)
python3 - <<PY
start, end = float("$START"), float("$END")
print(f"wall: {end - start:.1f}s")
PY
decode_webp /tmp/klein-warm.json /tmp/klein-warm.webp

echo "=== describe ==="
curl -sS -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d '{"path": "__DESCRIBE__"}' \
  | python3 -m json.tool | head -40
