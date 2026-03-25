#!/usr/bin/env bash
# scripts/dev.sh — start the full dev stack
set -euo pipefail

cd "$(dirname "$0")/.."   # always run from repo root

echo "▶ Starting dev stack…"
docker compose -f docker-compose.dev.yml up --build "$@"
