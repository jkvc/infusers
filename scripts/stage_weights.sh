#!/usr/bin/env bash
# Stage Klein 9B weights for Modal Volume upload.
#
# Layout: weights/klein-9b/ — gitignored.
#
# Usage: ./scripts/stage_weights.sh
set -euo pipefail

cd "$(dirname "$0")/.."

resolve_hf_blob() {
  local glob_path="$1"
  local resolved
  resolved=$(readlink -f "$glob_path")
  if [[ ! -f "$resolved" ]]; then
    echo "Missing local weight: $glob_path" >&2
    exit 1
  fi
  printf '%s' "$resolved"
}

root="weights/klein-9b"
ckpt_dir="$root/klein-9b"
hf_home="$root/hf"

mkdir -p "$ckpt_dir" "$hf_home"

echo "==> Copying Klein 9B flow + VAE from local HF cache ..."
klein_wt=$(resolve_hf_blob "$HOME/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9B/snapshots/"*/flux-2-klein-9b.safetensors)
ae_wt=$(resolve_hf_blob "$HOME/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots/"*/ae.safetensors)

cp -L "$klein_wt" "$ckpt_dir/flux-2-klein-9b.safetensors"
cp -L "$ae_wt" "$ckpt_dir/ae.safetensors"

echo "==> Downloading Qwen3-8B-FP8 text encoder into staged HF_HOME ..."
HF_HOME="$hf_home" uv run hf download Qwen/Qwen3-8B-FP8

echo
du -sh "$root" "$ckpt_dir" "$hf_home"
echo
echo "Staged under $root/"
echo "Next: ./scripts/upload_weights.sh"
