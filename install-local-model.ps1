[CmdletBinding()]
param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRoot = Join-Path $ProjectRoot "backend"
$VenvPython = Join-Path $BackendRoot ".venv\Scripts\python.exe"
$TorchVersion = "2.7.1"
$PytorchHost = "download.pytorch.org"

function Install-PytorchRuntime {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Package,

        [Parameter(Mandatory = $true)]
        [string]$IndexUrl
    )

    $PipArguments = @(
        "-m", "pip", "install",
        $Package,
        "--index-url", $IndexUrl,
        "--timeout", "180",
        "--retries", "10",
        "--no-cache-dir"
    )

    Write-Host "从 PyTorch 官方索引安装：$Package" -ForegroundColor Cyan
    & $VenvPython @PipArguments
    if ($LASTEXITCODE -eq 0) {
        return
    }

    Write-Host "标准 TLS 下载未完成，自动启用官方域名容错重试。" -ForegroundColor Yellow
    Write-Host "此放宽仅作用于本次 pip 命令和 download.pytorch.org。" -ForegroundColor DarkGray
    & $VenvPython @PipArguments --trusted-host $PytorchHost
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch 下载仍然失败。请切换网络后重试；也可以先执行 .\setup.ps1 -Mode api，稍后再安装本地模型。"
    }
}

if (-not (Test-Path $VenvPython)) {
    throw "尚未创建 RAG 后端虚拟环境，请先执行 .\setup.ps1"
}

if ($Device -eq "auto") {
    $Device = if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) {
        "cuda"
    }
    else {
        "cpu"
    }
}

$TorchFlavor = & $VenvPython -c "import importlib.util as u; s=u.find_spec('torch'); print('missing' if s is None else ('cuda' if __import__('torch').version.cuda else 'cpu'))"
if ($LASTEXITCODE -ne 0) {
    $TorchFlavor = "broken"
}

if ($Device -eq "cuda" -and $TorchFlavor -ne "cuda") {
    Write-Host "安装 NVIDIA CUDA 版 PyTorch $TorchVersion（cu126）" -ForegroundColor Cyan
    if ($TorchFlavor -ne "missing") {
        & $VenvPython -m pip uninstall -y torch
        if ($LASTEXITCODE -ne 0) { throw "移除不匹配的 PyTorch 失败。" }
    }
    Install-PytorchRuntime `
        -Package "torch==$TorchVersion+cu126" `
        -IndexUrl "https://download.pytorch.org/whl/cu126"
}
elseif ($Device -eq "cpu" -and $TorchFlavor -in @("missing", "broken")) {
    Write-Host "安装 CPU 版 PyTorch $TorchVersion" -ForegroundColor Cyan
    if ($TorchFlavor -eq "broken") {
        & $VenvPython -m pip uninstall -y torch
    }
    Install-PytorchRuntime `
        -Package "torch==$TorchVersion+cpu" `
        -IndexUrl "https://download.pytorch.org/whl/cpu"
}
else {
    Write-Host "复用当前 RAG 虚拟环境中的 PyTorch：$TorchFlavor" -ForegroundColor DarkGray
}

Write-Host "安装 Transformers、Accelerate、PEFT 与 Safetensors" -ForegroundColor Cyan
& $VenvPython -m pip install `
    -r (Join-Path $BackendRoot "requirements-local-model.txt") `
    --timeout 180 `
    --retries 10
if ($LASTEXITCODE -ne 0) { throw "安装本地模型依赖失败。" }

& $VenvPython -c "import torch, transformers, accelerate, peft, safetensors; print('Torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('Transformers:', transformers.__version__); print('PEFT:', peft.__version__)"
if ($LASTEXITCODE -ne 0) { throw "本地模型依赖导入失败。" }

if ($Device -eq "cuda") {
    & $VenvPython -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 2)"
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA 版 PyTorch 已安装，但当前进程仍无法访问 GPU。请检查 NVIDIA 驱动，并确认没有启动错虚拟环境。"
    }
}

Write-Host "本地模型运行环境已就绪。" -ForegroundColor Green
Write-Host "在 /settings 的主输入框中可直接填写完整模型目录或 outputs\run_xxx。" -ForegroundColor Cyan
