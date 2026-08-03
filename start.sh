#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_ROOT="$PROJECT_ROOT/runtime/logs"
mkdir -p "$LOG_ROOT"
[[ -x "$PROJECT_ROOT/backend/.venv/bin/python" && -d "$PROJECT_ROOT/frontend/node_modules" ]] || {
  echo "依赖尚未安装，请先执行 ./setup.sh" >&2
  exit 1
}

"$PROJECT_ROOT/start-backend.sh" >"$LOG_ROOT/backend.log" 2>&1 &
BACKEND_PID=$!
cleanup() { kill "$BACKEND_PID" "${FRONTEND_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

for _ in {1..40}; do
  curl -fsS "http://127.0.0.1:8000/api/v1/health" >/dev/null 2>&1 && break
  sleep 0.5
done
curl -fsS "http://127.0.0.1:8000/api/v1/health" >/dev/null || {
  echo "后端未就绪，请查看 runtime/logs/backend.log" >&2; exit 1;
}

"$PROJECT_ROOT/start-frontend.sh" >"$LOG_ROOT/frontend.log" 2>&1 &
FRONTEND_PID=$!
for _ in {1..80}; do
  curl -fsS "http://127.0.0.1:3000" >/dev/null 2>&1 && break
  sleep 0.5
done
curl -fsS "http://127.0.0.1:3000" >/dev/null || {
  echo "前端未就绪，请查看 runtime/logs/frontend.log" >&2; exit 1;
}

echo "RAG Studio 已就绪：http://localhost:3000"
open "http://localhost:3000"
echo "保持此终端开启；按 Control+C 停止前后端。"
wait "$FRONTEND_PID"
