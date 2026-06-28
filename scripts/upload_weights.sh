#!/usr/bin/env bash
# Upload staged weights to Modal Volume (one-time, ~27GB).
#
# Prereqs: uv run modal setup, ./scripts/stage_weights.sh
#
# Usage: ./scripts/upload_weights.sh
set -euo pipefail

VOLUME=jkvc-klein-9b-weights
LOCAL=weights/klein-9b
REMOTE=klein-9b

cd "$(dirname "$0")/.."

if [[ ! -d "$LOCAL/klein-9b" ]]; then
  echo "Missing $LOCAL/klein-9b — run ./scripts/stage_weights.sh first" >&2
  exit 1
fi

echo "Creating volume $VOLUME (if needed)..."
uv run modal volume create "$VOLUME" 2>/dev/null || true

echo "Uploading $(du -sh "$LOCAL" | cut -f1) from $LOCAL -> /$REMOTE ..."
uv run modal volume put "$VOLUME" "$LOCAL" "/$REMOTE"

echo "Done. Verify:"
uv run modal volume ls "$VOLUME" "/$REMOTE"
