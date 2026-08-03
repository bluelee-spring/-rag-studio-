#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$PROJECT_ROOT/backend/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "尚未创建后端虚拟环境，请先执行 ./setup.sh" >&2
  exit 1
fi

echo "安装 macOS PyTorch（Apple Silicon 将自动获得 MPS 支持）"
"$PYTHON" -m pip install "torch==2.7.1" --timeout 180 --retries 10 --no-cache-dir
"$PYTHON" -m pip install -r "$PROJECT_ROOT/backend/requirements-local-model.txt" --timeout 180 --retries 10
"$PYTHON" - <<'PY'
import torch, transformers, accelerate, peft, safetensors
print("Torch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
print("Device:", "mps" if torch.backends.mps.is_available() else "cpu")
print("Transformers:", transformers.__version__)
print("PEFT:", peft.__version__)
PY
echo "本地模型运行环境已就绪。"
