#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"
ENV_EXAMPLE="$PROJECT_DIR/.env.example"

if ! command -v openssl >/dev/null 2>&1; then
    echo "ERROR: openssl is required to generate deployment secrets." >&2
    echo "Run 'make bootstrap' or install openssl with your system package manager." >&2
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    if [ ! -f "$ENV_EXAMPLE" ]; then
        echo "ERROR: $ENV_EXAMPLE is missing." >&2
        exit 1
    fi
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    echo "[INFO] Created .env from .env.example."
fi

set_env_secret() {
    local key="$1"
    local value
    value="$(openssl rand -hex 32)"

    if grep -q "^${key}=" "$ENV_FILE"; then
        if grep -q "^${key}=$" "$ENV_FILE"; then
            sed -i "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
            echo "[OK] Generated ${key}."
        else
            echo "[INFO] Preserved existing ${key}."
        fi
    else
        printf '\n%s=%s\n' "$key" "$value" >> "$ENV_FILE"
        echo "[OK] Added ${key}."
    fi
}

set_env_secret DEEP_VQA_API_KEY
set_env_secret DEEP_VQA_AUTH_SECRET
echo "[OK] Deployment secrets are ready in .env."
