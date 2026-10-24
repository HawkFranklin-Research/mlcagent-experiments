#!/usr/bin/env bash
# Pool the cloud results on the laptop: download every archive of one run tag from the private
# Hugging Face dataset, verify its SHA-256 and unpack it next to the other run folders.
#
# Usage: scripts/cloud/fetch_cloud_runs.sh RUN_TAG DEST_PARENT
#   e.g. scripts/cloud/fetch_cloud_runs.sh eval-02 bio-bench/runs/deepseek-flash
# An existing run folder is never overwritten. Token: MLCA_HF_TOKEN_FILE
# (default ~/Documents/hugging-face-read-token.txt), passed only through the environment.
set -euo pipefail

[ $# -eq 2 ] || { echo "Usage: $0 RUN_TAG DEST_PARENT" >&2; exit 2; }
TAG="$1"
DEST="$(realpath "$2")"
HF_REPO="HawkFranklin-Research/mlcagent-runs"
TOKEN_FILE="${MLCA_HF_TOKEN_FILE:-$HOME/Documents/hugging-face-read-token.txt}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

HF_TOKEN="$(tr -d '\r\n[:space:]' <"$TOKEN_FILE")" \
    uvx --quiet --from huggingface_hub==1.13.0 hf download "$HF_REPO" --repo-type dataset \
    --include "cloud/$TAG/*" --local-dir "$WORK" >/dev/null

shopt -s nullglob
found=0
for archive in "$WORK"/cloud/"$TAG"/*/*.tar.zst; do
    found=1
    host="$(basename "$(dirname "$archive")")"
    (cd "$(dirname "$archive")" && sha256sum -c --quiet "$(basename "$archive").sha256") \
        || { echo "CHECKSUM MISMATCH: $host/$(basename "$archive")" >&2; exit 1; }
    name="$(zstd -dc "$archive" | tar -t | head -1 | cut -d/ -f1)"
    if [ -e "$DEST/$name" ]; then
        echo "kept existing $DEST/$name (from $host; not overwritten)"
        continue
    fi
    zstd -dc "$archive" | tar -x -C "$DEST"
    echo "unpacked $name from $host ($(cut -d' ' -f1 "$archive.sha256"))"
done
[ "$found" = 1 ] || { echo "No archives under cloud/$TAG on $HF_REPO" >&2; exit 1; }
