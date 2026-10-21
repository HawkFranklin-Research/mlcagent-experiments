#!/usr/bin/env bash
# One local diagnostic study; no Codespaces, no existing-result resume or cleanup.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$project_root/../mlca-revized/.venv/bin/python"
scratch="$HOME/.cache/mlca-smoke-ollama"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--dry-run" ) ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi
# A dry run reads no credentials, creates no files and starts no study.
if [[ ${1:-} == "--dry-run" ]]; then
    output="$scratch/FRESH-SESSION/run-01"
else
    mkdir -p -- "$scratch"
    session="$(mktemp -d "$scratch/session-XXXXXXXX")"
    output="$session/run-01"
fi
command=("$python_bin" "$project_root/scripts/run_batch.py"
    --out "$output" --batch 900 --tables breast_cancer_diagnostic --replicates 1
    --provider ollama --model minimax-m3.1-flash --structured-outputs on
    --reasoning-effort low --sdk-timeout-s 300 --model-call-deadline-s 600
    --concurrency 1 --threads 2 --max-output-tokens 32768 --rpm 20)
if [[ ${1:-} == "--dry-run" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
if [[ -z ${OLLAMA_API_KEY:-} ]]; then
    export OLLAMA_API_KEY="$(cat ~/Documents/ollama-mlca-api.txt)"
fi
export PYTHONPATH="$project_root/../mlca-revized/src${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$project_root"
# Retain output and its failure evidence. The smoke owner reviews before cleanup.
exec "${command[@]}"
