# RAG Studio v0.6.2

本版本增加完整的 macOS 安装与运行支持。

- 新增 `setup.sh`：支持在线 API 和本地模型两种安装方式。
- 新增 `start.sh`：启动并检查 FastAPI、Next.js，随后打开浏览器。
- 新增独立的前端、后端、本地模型安装和环境检查脚本。
- 同时兼容 Apple Silicon 与 Intel Mac。
- Apple Silicon 本地推理支持自动选择 PyTorch MPS；不支持 MPS 时回退 CPU。
- 模型设置页面新增 Apple MPS 设备选项。

macOS 首次运行：`chmod +x ./*.sh && ./setup.sh`。以后运行：`./start.sh`。
