#!/usr/bin/env bash
# Run one VM's share of the AutoML comparison (AutoGluon and FLAML on MLCA's frozen splits; no LLM),
# back the results up to the private Hugging Face dataset, then power the VM off.
#
# Usage (inside tmux, from the experiments repository root):
#   scripts/cloud/run_automl_vm.sh scripts/cloud/slices/automl-bio-vmN.sh
#
# The slice file defines:
#   RUN_TAG="automl-bio"                      # Hugging Face folder: cloud/<RUN_TAG>/<host>/
#   OUT="bench/automl-bio"                    # result root, shared by all VMs (disjoint task folders)
#   SPEC="automl/specs/biomedical.yaml"
#   COMMON=(--time-budget 600 --cpus 2 --seed 42)
#   LANES=("<MLCA batch folder>|<task id> [<task id> …]" …)   # lanes run in parallel
#
# Existing result.json files (including failures) are never rerun, so the same line resumes after a
# Spot preemption. The private partitions under $OUT/.work are never uploaded. Set MLCA_NO_POWEROFF=1
# to keep the VM running. Hugging Face write token: MLCA_HF_TOKEN_FILE
# (default ~/Documents/hugging-face-write-token.txt), passed only through the environment.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
HF_REPO="HawkFranklin-Research/mlcagent-runs"
HF_TOKEN_FILE="${MLCA_HF_TOKEN_FILE:-$HOME/Documents/hugging-face-write-token.txt}"
HOST="$(hostname)"
LOG="$ROOT/cloud-automl-$HOST.log"

[ $# -eq 1 ] && [ -f "$1" ] || { echo "Usage: $0 SLICE_FILE" >&2; exit 2; }
# shellcheck source=/dev/null
source "$1"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

[ -x automl/.venv/bin/python ] || { log "automl/.venv missing (see automl/README.md); nothing started"; exit 2; }

pids=()
for lane in "${LANES[@]}"; do
    runs="${lane%%|*}"
    read -r -a tasks <<<"${lane#*|}"
    [ -d "$runs/studies" ] || { log "missing MLCA batch folder $runs; lane skipped"; continue; }
    log "lane start: $runs ${tasks[*]}"
    (
        PYTHONPATH=../mlca-revized/src automl/.venv/bin/python automl/run_automl.py --batch \
            --spec "$SPEC" --tasks "${tasks[@]}" --runs-root "$runs" \
            --mlca-python ../mlca-revized/.venv/bin/python "${COMMON[@]}" \
            --work-root "$OUT/.work" --out "$OUT" >>"$LOG" 2>&1
        echo "lane exit $?: $runs ${tasks[*]}" >>"$LOG"
    ) &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
automl/.venv/bin/python automl/summarize.py --out "$OUT" >>"$LOG" 2>&1
log "lanes finished; results under $OUT"

name="$(basename "$OUT")"
archive="/tmp/$name-$HOST.tar.zst"
if grep --exclude-dir=.work -rIlE 'sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|gh[ps]_[A-Za-z0-9]{20,}' "$OUT" >/dev/null 2>&1; then
    log "SECRET PATTERN FOUND in $OUT; not uploaded"
elif [ ! -s "$HF_TOKEN_FILE" ]; then
    log "HF token file missing; results stay on disk"
else
    # .work holds the private train/validation/test partitions: never archived.
    tar --exclude='.work' -C "$(dirname "$OUT")" -cf - "$name" | zstd -q -T0 -19 -o "$archive" -f
    sha256sum "$archive" | sed "s#/tmp/##" >"$archive.sha256"
    for f in "$archive" "$archive.sha256"; do
        HF_TOKEN="$(tr -d '\r\n[:space:]' <"$HF_TOKEN_FILE")" \
            uvx --quiet --from huggingface_hub==1.13.0 hf upload "$HF_REPO" "$f" \
            "cloud/$RUN_TAG/$HOST/$(basename "$f")" --repo-type dataset \
            --commit-message "AutoML $RUN_TAG $HOST $(basename "$f")" >>"$LOG" 2>&1 \
            || log "upload failed: $f (kept on disk)"
    done
    log "uploaded $name: $(cut -d' ' -f1 "$archive.sha256")"
fi

log "slice finished on $HOST"
if [ -z "${MLCA_NO_POWEROFF:-}" ]; then
    log "powering off"
    sudo poweroff
fi
