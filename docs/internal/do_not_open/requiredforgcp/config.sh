#!/usr/bin/env bash
# Shared config. Every other script does:  source config.sh
# It loads your .env and derives a few values. Do not run directly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
  echo "ERROR: $SCRIPT_DIR/.env not found."
  echo "       Run:  cp .env.example .env   and fill it in."
  exit 1
fi

# Export every var defined in .env
set -a
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env"
set +a

# Sanity checks
: "${PROJECT_ID:?set PROJECT_ID in .env}"
: "${ZONE:?set ZONE in .env}"
: "${VM_NAME:?set VM_NAME in .env}"
: "${BUCKET:?set BUCKET in .env}"
: "${HF_TOKEN:?set HF_TOKEN in .env}"
: "${API_KEY:?set API_KEY in .env}"

# Region is the zone without its trailing letter (us-central1-a -> us-central1)
export REGION="${ZONE%-*}"
export DEPLOY_DIR="$SCRIPT_DIR"

echo "[config] project=$PROJECT_ID zone=$ZONE region=$REGION vm=$VM_NAME"
