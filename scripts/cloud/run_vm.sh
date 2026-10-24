#!/usr/bin/env bash
# Run one VM's slice of a parallel batch, back up the results, then power the VM off.
#
# Usage (inside tmux, from the experiments repository root):
#   scripts/cloud/run_vm.sh scripts/cloud/slices/<slice>.sh
#
# The slice file is a bash file that defines:
#   RUN_TAG="eval-02"                              # folder on Hugging Face: cloud/<RUN_TAG>/<host>/
#   COMMON=(--role evaluation ... --rpm 60)       # identical on every VM
#   BATCHES=("bio-bench/runs/x/eval-02-vm1|--tables s2_rna_her2 --replicates 3" ...)
#
# Each batch starts fresh, or resumes if its batch.json exists (after a Spot preemption or a stop).
# Resume re-queues attempts lost with the machine automatically; it never retries other failures.
# If the controller pauses for the operator (infrastructure or provider error), later batches are
# not started. In every case the folders are archived to the private Hugging Face dataset and the
# VM powers off. Set MLCA_NO_POWEROFF=1 to keep it running. MLCA_RESUME_ARGS and MLCA_UPGRADE_REASON add flags to a resume.
#
# Credentials are read from files and passed only through the environment, never as arguments:
#   DeepSeek: MLCA_DEEPSEEK_KEY_FILE (default ~/Documents/cheap-seek.api-key.txt), via with_deepseek_key.sh
#   Hugging Face write: MLCA_HF_TOKEN_FILE (default ~/Documents/hugging-face-write-token.txt)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/../mlca-revized/.venv/bin/python"
HF_REPO="HawkFranklin-Research/mlcagent-runs"
HF_TOKEN_FILE="${MLCA_HF_TOKEN_FILE:-$HOME/Documents/hugging-face-write-token.txt}"
HOST="$(hostname)"
LOG="$ROOT/cloud-run-$HOST.log"

[ $# -eq 1 ] && [ -f "$1" ] || { echo "Usage: $0 SLICE_FILE" >&2; exit 2; }
# shellcheck source=/dev/null
source "$1"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

status_of() {
    "$PY" - "$1" <<'EOF'
import json, sys
from pathlib import Path
f = Path(sys.argv[1]) / "batch.json"
if not f.exists():
    print("new")
else:
    s = json.loads(f.read_text())
    done = all(j["status"] == "finished" for j in s["jobs"])
    print("complete" if done else s.get("status") or "unfinished")
EOF
}

scripts/with_deepseek_key.sh --check >>"$LOG" 2>&1 || { log "DeepSeek key missing; nothing started"; exit 2; }

stopped_early=""
for item in "${BATCHES[@]}"; do
    out="${item%%|*}"
    read -r -a extra <<<"${item#*|}"
    state="$(status_of "$out")"
    if [ "$state" = "complete" ]; then
        log "$out already complete"
        continue
    fi
    mode=()
    # On a resume only: MLCA_RESUME_ARGS adds flags (no spaces inside a flag, e.g.
    # --retry-infrastructure-failures) and MLCA_UPGRADE_REASON adds --controller-upgrade "<reason>".
    if [ "$state" != "new" ]; then
        read -r -a mode <<<"--resume ${MLCA_RESUME_ARGS:-}"
        [ -z "${MLCA_UPGRADE_REASON:-}" ] || mode+=(--controller-upgrade "$MLCA_UPGRADE_REASON")
    fi
    log "start $out ($state) ${mode[*]} ${extra[*]}"
    scripts/with_deepseek_key.sh "$PY" scripts/run_batch.py "${COMMON[@]}" --out "$out" "${extra[@]}" "${mode[@]}" 2>&1 | tee -a "$LOG"
    state="$(status_of "$out")"
    log "end $out: $state"
    if [ "$state" != "complete" ]; then
        stopped_early="$out ($state)"
        break
    fi
done

backup() {
    local out="$1" name archive
    name="$(basename "$out")"
    archive="/tmp/$name-$HOST.tar.zst"
    if grep -rIlE 'sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|gh[ps]_[A-Za-z0-9]{20,}' "$out" >/dev/null 2>&1; then
        log "SECRET PATTERN FOUND in $out; not uploaded"
        return 1
    fi
    tar --exclude='.control/rate' --exclude='.control/config' -C "$(dirname "$out")" -cf - "$name" | zstd -q -T0 -19 -o "$archive" -f
    sha256sum "$archive" | sed "s#/tmp/##" >"$archive.sha256"
    [ -s "$HF_TOKEN_FILE" ] || { log "HF token file missing; $archive kept on disk"; return 1; }
    for f in "$archive" "$archive.sha256"; do
        HF_TOKEN="$(tr -d '\r\n[:space:]' <"$HF_TOKEN_FILE")" \
            uvx --quiet --from huggingface_hub==1.13.0 hf upload "$HF_REPO" "$f" \
            "cloud/$RUN_TAG/$HOST/$(basename "$f")" --repo-type dataset \
            --commit-message "Cloud run $RUN_TAG $HOST $(basename "$f")" >>"$LOG" 2>&1 \
            || { log "upload failed: $f (kept on disk)"; return 1; }
    done
    log "uploaded $name: $(cut -d' ' -f1 "$archive.sha256")"
}

for item in "${BATCHES[@]}"; do
    out="${item%%|*}"
    [ -f "$out/batch.json" ] && backup "$out"
done

if [ -n "$stopped_early" ]; then
    log "OPERATOR NEEDED: stopped at $stopped_early; see its REPORT.md"
fi
log "slice finished on $HOST"
if [ -z "${MLCA_NO_POWEROFF:-}" ]; then
    log "powering off"
    sudo poweroff
fi
