#!/usr/bin/env bash
# scripts/prod.sh — build and start the production stack (detached)
set -euo pipefail

cd "$(dirname "$0")/.."

echo "▶ Building and starting prod stack…"
docker compose -f docker-compose.prod.yml up --build -d "$@"
echo "✔ Running at http://localhost"
