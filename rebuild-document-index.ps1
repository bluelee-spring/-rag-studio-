$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRoot = Join-Path $ProjectRoot "backend"
$VenvPython = Join-Path $BackendRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    throw "尚未安装后端依赖，请先执行 .\setup.ps1"
}

Set-Location $BackendRoot
& $VenvPython -m app.scripts.build_document_index --force
