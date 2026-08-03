$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendScript = Join-Path $ProjectRoot "start-backend.ps1"
$FrontendScript = Join-Path $ProjectRoot "start-frontend.ps1"
$VenvPython = Join-Path $ProjectRoot "backend\.venv\Scripts\python.exe"
$NodeModules = Join-Path $ProjectRoot "frontend\node_modules"

if (-not (Test-Path $VenvPython) -or -not (Test-Path $NodeModules)) {
    throw "依赖尚未安装，请先执行 .\setup.ps1"
}

Write-Host "启动 FastAPI 后端……" -ForegroundColor Cyan
$BackendArguments = "-NoExit -ExecutionPolicy Bypass -File `"$BackendScript`""
Start-Process powershell.exe -ArgumentList $BackendArguments | Out-Null

$BackendReady = $false
for ($Attempt = 0; $Attempt -lt 40; $Attempt++) {
    try {
        $Response = Invoke-WebRequest "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 1
        if ($Response.StatusCode -eq 200) {
            $BackendReady = $true
            break
        }
    }
    catch {
        Start-Sleep -Milliseconds 500
    }
}
if (-not $BackendReady) {
    throw "FastAPI 在 20 秒内没有就绪，请查看新打开的后端 PowerShell 窗口。"
}

Write-Host "启动 Next.js 前端……" -ForegroundColor Cyan
$FrontendArguments = "-NoExit -ExecutionPolicy Bypass -File `"$FrontendScript`""
Start-Process powershell.exe -ArgumentList $FrontendArguments | Out-Null

$FrontendReady = $false
for ($Attempt = 0; $Attempt -lt 80; $Attempt++) {
    try {
        $Response = Invoke-WebRequest "http://localhost:3000" -UseBasicParsing -TimeoutSec 1
        if ($Response.StatusCode -eq 200) {
            $FrontendReady = $true
            break
        }
    }
    catch {
        Start-Sleep -Milliseconds 500
    }
}
if (-not $FrontendReady) {
    throw "Next.js 在 40 秒内没有就绪，请查看新打开的前端 PowerShell 窗口。"
}

Write-Host "RAG Studio 已就绪：http://localhost:3000" -ForegroundColor Green
Start-Process "http://localhost:3000"
