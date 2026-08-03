[CmdletBinding()]
param(
    [ValidateSet("ask", "api", "cpu", "cuda")]
    [string]$Mode = "ask"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRoot = Join-Path $ProjectRoot "backend"
$FrontendRoot = Join-Path $ProjectRoot "frontend"
$VenvRoot = Join-Path $BackendRoot ".venv"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"
$RuntimeRoot = Join-Path $ProjectRoot "runtime"

Write-Host "RAG Studio 首次安装" -ForegroundColor Green
Write-Host "项目目录：$ProjectRoot" -ForegroundColor DarkGray

if ($Mode -eq "ask") {
    Write-Host ""
    Write-Host "请选择模型运行方式：" -ForegroundColor Cyan
    Write-Host "  [1] 在线 API       不安装 PyTorch，安装最快"
    Write-Host "  [2] 本地模型 CPU   可离线生成，但速度较慢"
    Write-Host "  [3] 本地模型 CUDA  NVIDIA GPU，推荐"
    $Choice = Read-Host "请输入 1、2 或 3"
    $Mode = switch ($Choice) {
        "1" { "api" }
        "2" { "cpu" }
        "3" { "cuda" }
        default { throw "无效选择：$Choice" }
    }
}

Write-Host "[1/6] 检查 Python" -ForegroundColor Cyan
$PythonCommand = (Get-Command python.exe -ErrorAction Stop).Source
$PythonVersion = & $PythonCommand -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
if ($LASTEXITCODE -ne 0) {
    throw "无法运行 Python。"
}
$PythonSupported = & $PythonCommand -c "import sys; print('yes' if (3, 10) <= sys.version_info[:2] <= (3, 12) else 'no')"
if ($PythonSupported -ne "yes") {
    throw "当前 Python 为 $PythonVersion。本项目支持 Python 3.10–3.12，推荐 Python 3.11。"
}
Write-Host "Python $PythonVersion" -ForegroundColor DarkGray

Write-Host "[2/6] 创建项目独立虚拟环境" -ForegroundColor Cyan
if (-not (Test-Path $VenvPython)) {
    & $PythonCommand -m venv $VenvRoot
    if ($LASTEXITCODE -ne 0) {
        throw "创建 backend\.venv 失败。"
    }
}
New-Item -ItemType Directory -Path (Join-Path $RuntimeRoot "workspaces") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $RuntimeRoot "ingestion") -Force | Out-Null

Write-Host "[3/6] 安装 FastAPI、检索与文件处理依赖" -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "升级 pip 失败。" }
& $VenvPython -m pip install -r (Join-Path $BackendRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "安装后端基础依赖失败。" }

$BackendEnv = Join-Path $BackendRoot ".env"
if (-not (Test-Path $BackendEnv)) {
    Copy-Item (Join-Path $BackendRoot ".env.example") $BackendEnv
    Write-Host "已创建 backend\.env。API Key、Neo4j 密码只在本机填写。" -ForegroundColor Yellow
}

Write-Host "[4/6] 配置生成模型运行环境：$Mode" -ForegroundColor Cyan
if ($Mode -in @("cpu", "cuda")) {
    & (Join-Path $ProjectRoot "install-local-model.ps1") -Device $Mode
    if ($LASTEXITCODE -ne 0) {
        throw "本地模型运行环境安装失败。"
    }
}
else {
    Write-Host "在线 API 模式不安装 PyTorch；以后可执行 .\install-local-model.ps1 增加本地推理。" -ForegroundColor DarkGray
}

Write-Host "[5/6] 构建内置文档 FAISS 索引" -ForegroundColor Cyan
Push-Location $BackendRoot
try {
    & $VenvPython -m app.scripts.build_document_index
    if ($LASTEXITCODE -ne 0) { throw "构建内置 FAISS 索引失败。" }
}
finally {
    Pop-Location
}

Write-Host "[6/6] 安装 Next.js 前端依赖" -ForegroundColor Cyan
Get-Command npm.cmd -ErrorAction Stop | Out-Null
Push-Location $FrontendRoot
try {
    if (Test-Path (Join-Path $FrontendRoot "package-lock.json")) {
        & npm.cmd ci
    }
    else {
        & npm.cmd install
    }
    if ($LASTEXITCODE -ne 0) { throw "安装前端依赖失败。" }
}
finally {
    Pop-Location
}

if ($Mode -in @("cpu", "cuda")) {
    & (Join-Path $ProjectRoot "check-environment.ps1") -RequireLocalModel
}
else {
    & (Join-Path $ProjectRoot "check-environment.ps1")
}
if ($LASTEXITCODE -ne 0) {
    throw "环境自检未通过。"
}

Write-Host ""
Write-Host "安装完成。以后只需执行 .\start.ps1" -ForegroundColor Green
Write-Host "模型配置：http://localhost:3000/settings" -ForegroundColor Cyan
Write-Host "数据工作区：http://localhost:3000/data" -ForegroundColor Cyan
