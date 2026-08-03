# RAG Teaching Studio：Windows 与 macOS 运行讲义

## 1. 环境要求(完整跑过Lora微调的同学环境都是适应的)

安装并确认：

- Python 3.10–3.12，推荐 Python 3.11。
- Node.js 20.9 或更高版本。
- 解压后的项目目录中能看到 `setup.ps1`、`setup.sh`、`start.ps1`、`start.sh`。

不要把 LoRA 项目的虚拟环境复制到 RAG 项目。安装脚本会创建：

```text
rag-teaching-studio/backend/.venv
```

## 2. Windows 安装

### 2.1 检查系统环境

打开 PowerShell：

```powershell
python --version
node --version
npm --version
```

Python 必须为 3.10–3.12，Node.js 必须为 20.9 或更高版本。

### 2.2 解压并进入项目根目录（如果解压过了，直接进入rag-teaching-studio这个目录里）

```powershell
Expand-Archive `
  -Path "D:\Downloads\RAG-Teaching-Studio-v0.7.0-Graph-Workspace.zip" `
  -DestinationPath "D:\RAG-Studio"

Set-Location "D:\RAG-Studio\rag-teaching-studio"

Test-Path ".\setup.ps1"
```

最后一条命令必须返回：

```text
True
```

### 2.3 选择一种安装方式

使用在线 API：（最快，但是需要配置大模型的API，API_URL和API_key）

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1 -Mode api
```

使用 NVIDIA GPU 本地模型：(不推荐，可能存在cuda与torch兼容的问题)

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1 -Mode cuda
```

没有 NVIDIA GPU，使用 CPU 本地模型：（推荐）

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1 -Mode cpu
```

三种方式只选一种。首次安装需要等待 Python 和 npm 依赖下载完成。

### 2.4 启动

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\start.ps1
```

打开：

```text
实验台：http://localhost:3000
后端文档：http://127.0.0.1:8000/docs
```

## 3. macOS 安装

### 3.1 检查系统环境

打开 Terminal：

```bash
python3 --version
node --version
npm --version
```

Python 必须为 3.10–3.12，Node.js 必须为 20.9 或更高版本。

如果 `python3` 不是正确版本，但已经安装 `python3.11`，后续命令使用：

```bash
PYTHON_COMMAND=python3.11 ./setup.sh api
```

或：

```bash
PYTHON_COMMAND=python3.11 ./setup.sh local
```

### 3.2 解压并进入项目根目录

```bash
mkdir -p "$HOME/RAG-Studio"

unzip "$HOME/Downloads/RAG-Teaching-Studio-v0.7.0-Graph-Workspace.zip" \
  -d "$HOME/RAG-Studio"

cd "$HOME/RAG-Studio/rag-teaching-studio"

chmod +x ./*.sh

test -f ./setup.sh && echo "项目目录正确"
```

### 3.3 选择一种安装方式

使用在线 API：

```bash
./setup.sh api
```

使用本地模型：

```bash
./setup.sh local
```

Apple Silicon Mac 优先使用 MPS；Intel Mac 或不支持 MPS 的机器使用 CPU。

### 3.4 启动

```bash
./start.sh
```

打开：

```text
实验台：http://localhost:3000
后端文档：http://127.0.0.1:8000/docs
```

保持 Terminal 开启。停止系统时按：

```text
Control + C
```

## 4. 配置模型

打开：

```text
http://localhost:3000/settings
```

### 4.1 在线 API

选择“远程 API”，填写：

```text
API Base
API Key
生成模型
Embedding 模型（可选）
```

点击“保存并测试连接”。

### 4.2 本地完整模型

模型目录必须存在于运行 RAG 后端的同一台电脑。

Windows 示例：

```text
C:\Users\用户名\Downloads\LoRA-Visual-Lab\lora-visual-lab\backend\models\Qwen2.5-0.5B-Instruct
```

macOS 示例：

```text
/Users/用户名/Downloads/LoRA-Visual-Lab/lora-visual-lab/backend/models/Qwen2.5-0.5B-Instruct
```

前端选择：

```text
模型类型：本地 Hugging Face / LoRA
设备：自动检测
精度：自动检测
```

点击“保存并加载测试”。

### 4.3 LoRA 微调输出

可以直接填写 LoRA 运行目录：

Windows：

```text
C:\Users\用户名\Downloads\LoRA-Visual-Lab\lora-visual-lab\backend\outputs\run_xxx
```

macOS：

```text
/Users/用户名/Downloads/LoRA-Visual-Lab/lora-visual-lab/backend/outputs/run_xxx
```

该目录至少应包含：

```text
adapter_config.json
adapter_model.safetensors
```

## 5. 环境自检

### Windows

基础功能：

```powershell
.\check-environment.ps1
```

包含本地模型功能：

```powershell
.\check-environment.ps1 -RequireLocalModel
```

后端健康检查：

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/health"
```

### macOS

基础功能：

```bash
./check-environment.sh
```

包含本地模型功能：

```bash
./check-environment.sh --require-local-model
```

后端健康检查：

```bash
curl -fsS "http://127.0.0.1:8000/api/v1/health"
```

## 6. 常见错误

### CHECK FAILED：模型目录有效，但本地推理依赖不完整

原因：本地推理依赖没有安装到 RAG 项目的 `backend/.venv`。

Windows NVIDIA GPU：

```powershell
.\install-local-model.ps1 -Device cuda
.\check-environment.ps1 -RequireLocalModel
```

Windows CPU：

```powershell
.\install-local-model.ps1 -Device cpu
.\check-environment.ps1 -RequireLocalModel
```

macOS：

```bash
./install-local-model.sh
./check-environment.sh --require-local-model
```

安装完成后必须重启后端，再重新点击“保存并加载测试”。

### 无法识别 `setup.ps1` 或 `setup.sh`

当前目录错误。确认当前目录中存在启动脚本。

Windows：

```powershell
Get-Location
Get-ChildItem .\setup.ps1
```

macOS：

```bash
pwd
ls -l ./setup.sh
```

### Python 版本错误

项目只支持 Python 3.10–3.12。推荐 Python 3.11。不要使用 Python 3.9 或 3.13 创建本项目虚拟环境。

### 3000 或 8000 端口被占用

Windows 查看占用进程：

```powershell
Get-NetTCPConnection -State Listen -LocalPort 3000,8000 |
  Select-Object LocalPort,OwningProcess
```

macOS 查看占用进程：

```bash
lsof -nP -iTCP:3000 -sTCP:LISTEN
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

先关闭占用端口的旧 RAG 进程，再重新执行启动脚本。

## 7. 每次上课的启动命令

Windows：

```powershell
Set-Location "D:\RAG-Studio\rag-teaching-studio"
Set-ExecutionPolicy -Scope Process Bypass
.\start.ps1
```

macOS：

```bash
cd "$HOME/RAG-Studio/rag-teaching-studio"
./start.sh
```
