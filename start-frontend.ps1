$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$FrontendRoot = Join-Path $ProjectRoot "frontend"

if (-not (Test-Path (Join-Path $FrontendRoot "node_modules"))) {
    throw "尚未安装前端依赖，请先在项目根目录执行 .\setup.ps1"
}

Set-Location $FrontendRoot
$env:RAG_BACKEND_URL = if ($env:RAG_BACKEND_URL) {
    $env:RAG_BACKEND_URL
}
else {
    "http://127.0.0.1:8000"
}
& npm.cmd run dev -- --hostname 127.0.0.1
