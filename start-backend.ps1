$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRoot = Join-Path $ProjectRoot "backend"
$VenvPython = Join-Path $BackendRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    throw "尚未安装后端依赖，请先在项目根目录执行 .\setup.ps1"
}

Set-Location $BackendRoot
& $VenvPython -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
