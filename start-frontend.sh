#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
[[ -d "$PROJECT_ROOT/frontend/node_modules" ]] || { echo "请先执行 ./setup.sh" >&2; exit 1; }
cd "$PROJECT_ROOT/frontend"
export RAG_BACKEND_URL="${RAG_BACKEND_URL:-http://127.0.0.1:8000}"
exec npm run dev -- --hostname 127.0.0.1

