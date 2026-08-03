#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND_ROOT="$PROJECT_ROOT/backend"
FRONTEND_ROOT="$PROJECT_ROOT/frontend"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"
MODE="${1:-ask}"

if [[ "$MODE" == "ask" ]]; then
  echo "请选择模型运行方式："
  echo "  [1] 在线 API       不安装 PyTorch"
  echo "  [2] 本地模型       Apple Silicon 优先使用 MPS，Intel Mac 使用 CPU"
  read -r -p "请输入 1 或 2: " choice
  case "$choice" in
    1) MODE="api" ;;
    2) MODE="local" ;;
    *) echo "无效选择：$choice" >&2; exit 1 ;;
  esac
fi
[[ "$MODE" == "api" || "$MODE" == "local" ]] || { echo "用法：./setup.sh [api|local]" >&2; exit 1; }

command -v "$PYTHON_COMMAND" >/dev/null || { echo "未找到 Python 3.10–3.12" >&2; exit 1; }
command -v npm >/dev/null || { echo "未找到 npm，请先安装 Node.js 20.9+" >&2; exit 1; }
"$PYTHON_COMMAND" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
    raise SystemExit(f"需要 Python 3.10–3.12，当前为 {sys.version.split()[0]}")
print("Python:", sys.version.split()[0])
PY

if [[ ! -x "$BACKEND_ROOT/.venv/bin/python" ]]; then
  "$PYTHON_COMMAND" -m venv "$BACKEND_ROOT/.venv"
fi
PYTHON="$BACKEND_ROOT/.venv/bin/python"
mkdir -p "$PROJECT_ROOT/runtime/workspaces" "$PROJECT_ROOT/runtime/ingestion"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r "$BACKEND_ROOT/requirements.txt"
[[ -f "$BACKEND_ROOT/.env" ]] || cp "$BACKEND_ROOT/.env.example" "$BACKEND_ROOT/.env"

if [[ "$MODE" == "local" ]]; then
  "$PROJECT_ROOT/install-local-model.sh"
fi

(cd "$BACKEND_ROOT" && "$PYTHON" -m app.scripts.build_document_index)
(cd "$FRONTEND_ROOT" && npm ci)

if [[ "$MODE" == "local" ]]; then
  "$PROJECT_ROOT/check-environment.sh" --require-local-model
else
  "$PROJECT_ROOT/check-environment.sh"
fi
echo "安装完成。以后运行 ./start.sh"

