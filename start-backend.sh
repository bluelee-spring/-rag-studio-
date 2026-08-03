#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$PROJECT_ROOT/backend/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo "请先执行 ./setup.sh" >&2; exit 1; }
cd "$PROJECT_ROOT/backend"
exec "$PYTHON" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

