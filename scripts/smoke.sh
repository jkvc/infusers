#!/usr/bin/env bash
# HTTP smoke: Klein JSON endpoint — cold + warm infer + __DESCRIBE__.
#
# Setup (once):
#   cp .env.example .env
#   Fill MODAL_KEY / MODAL_SECRET from https://modal.com/settings/proxy-auth-tokens
#   Set MODAL_WEB_URL (from `uv run modal deploy infusers/modal_app/lunas_courageous_adventure.py`)
#
# Run:
#   ./scripts/smoke.sh
#
# Writes /tmp/klein-cold.webp and /tmp/klein-warm.webp.
# SSE stream: ./scripts/smoke_stream.sh
# No HTTP / no proxy token: uv run modal run infusers/modal_app/lunas_courageous_adventure.py::smoke
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/modal_env.sh"
_modal_env_load

URL="${MODAL_WEB_URL:?Set MODAL_WEB_URL in .env}"

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
import base64, json
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
  "${MODAL_AUTH_HEADERS[@]}" \
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
  "${MODAL_AUTH_HEADERS[@]}" \
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
  "${MODAL_AUTH_HEADERS[@]}" \
  -d '{"path": "__DESCRIBE__"}' \
  | python3 -m json.tool | head -40
