[CmdletBinding()]
param(
    [switch]$RequireLocalModel
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot "backend\.venv\Scripts\python.exe"
$FrontendRoot = Join-Path $ProjectRoot "frontend"

Write-Host "RAG Studio 环境自检" -ForegroundColor Green

if (-not (Test-Path $VenvPython)) {
    throw "缺少 backend\.venv，请执行 .\setup.ps1"
}

& $VenvPython -c "import sys; print('Python:', sys.executable); print('Version:', sys.version.split()[0])"
if ($LASTEXITCODE -ne 0) { throw "RAG 后端 Python 无法运行。" }

& $VenvPython -c "import importlib.util as u; names=['fastapi','uvicorn','pydantic','httpx','neo4j','rdflib','faiss','numpy','jieba','pypdf','docx','openpyxl']; missing=[n for n in names if u.find_spec(n) is None]; print('Backend core:', 'OK' if not missing else 'MISSING ' + ', '.join(missing)); raise SystemExit(1 if missing else 0)"
if ($LASTEXITCODE -ne 0) { throw "后端基础依赖不完整。" }

if ($RequireLocalModel) {
    & $VenvPython -c "import torch, transformers, accelerate, peft, safetensors; print('Local runtime: OK'); print('Torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('Transformers:', transformers.__version__); print('PEFT:', peft.__version__)"
    if ($LASTEXITCODE -ne 0) {
        throw "本地模型依赖不完整，请执行 .\install-local-model.ps1"
    }
}
else {
    & $VenvPython -c "import importlib.util as u; print('Local runtime:', 'INSTALLED' if u.find_spec('torch') and u.find_spec('transformers') else 'OPTIONAL / NOT INSTALLED')"
}

$NodeCommand = (Get-Command node.exe -ErrorAction Stop).Source
$NpmCommand = (Get-Command npm.cmd -ErrorAction Stop).Source
Write-Host "Node: $(& $NodeCommand --version)" -ForegroundColor DarkGray
Write-Host "npm: $(& $NpmCommand --version)" -ForegroundColor DarkGray

if (-not (Test-Path (Join-Path $FrontendRoot "node_modules"))) {
    throw "缺少 frontend\node_modules，请执行 .\setup.ps1"
}

Write-Host "环境自检通过。" -ForegroundColor Green
