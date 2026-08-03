#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$PROJECT_ROOT/backend/.venv/bin/python"
REQUIRE_LOCAL="${1:-}"

[[ -x "$PYTHON" ]] || { echo "缺少 backend/.venv，请执行 ./setup.sh" >&2; exit 1; }
command -v node >/dev/null || { echo "未找到 Node.js 20.9+" >&2; exit 1; }
command -v npm >/dev/null || { echo "未找到 npm" >&2; exit 1; }
[[ -d "$PROJECT_ROOT/frontend/node_modules" ]] || { echo "缺少前端依赖，请执行 ./setup.sh" >&2; exit 1; }

"$PYTHON" - <<'PY'
import importlib.util as u
import sys
names = ['fastapi','uvicorn','pydantic','httpx','neo4j','rdflib','faiss','numpy','jieba','pypdf','docx','openpyxl']
missing = [name for name in names if u.find_spec(name) is None]
print("Python:", sys.executable)
print("Backend core:", "OK" if not missing else "MISSING " + ", ".join(missing))
raise SystemExit(1 if missing else 0)
PY

if [[ "$REQUIRE_LOCAL" == "--require-local-model" ]]; then
  "$PYTHON" - <<'PY'
import torch, transformers, accelerate, peft, safetensors
print("Local runtime: OK")
print("MPS available:", torch.backends.mps.is_available())
PY
fi

echo "Node: $(node --version)"
echo "npm: $(npm --version)"
echo "环境自检通过。"

