#!/usr/bin/env bash
# scripts/with_deepseek_key.sh
# Load the author-approved local DeepSeek key, then exec a command. Never log it.
set -euo pipefail

KEY_FILE="${MLCA_DEEPSEEK_KEY_FILE:-$HOME/Documents/cheap-seek.api-key.txt}"

if [ "${1:-}" = "--check" ]; then
    if [ -n "${DEEPSEEK_API_KEY:-}" ]; then
        echo "DeepSeek credential available (existing environment; no API request made)."
        exit 0
    elif [ -f "$KEY_FILE" ] && [ -s "$KEY_FILE" ]; then
        echo "DeepSeek credential available (approved local key file: $KEY_FILE; no API request made)."
        exit 0
    else
        echo "Cannot load the approved DeepSeek key file; no credential value was logged." >&2
        exit 2
    fi
fi

if [ $# -eq 0 ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: scripts/with_deepseek_key.sh --check | COMMAND [ARGS...]"
    exit 0
fi

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    if [ ! -f "$KEY_FILE" ] || [ ! -s "$KEY_FILE" ]; then
        echo "Cannot load the approved DeepSeek key file; it must contain one plain-text key. No credential value was logged." >&2
        exit 2
    fi
    KEY="$(tr -d '\r\n[:space:]' < "$KEY_FILE")"
    if [ -z "$KEY" ]; then
        echo "Cannot load the approved DeepSeek key file; key is empty. No credential value was logged." >&2
        exit 2
    fi
    export DEEPSEEK_API_KEY="$KEY"
fi

exec "$@"
