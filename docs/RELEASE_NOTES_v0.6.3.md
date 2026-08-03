# RAG Studio v0.6.3

本版本将课堂中验证有效的 PyTorch 下载容错逻辑整合进 Windows 安装器。

- CPU 固定安装经过验证的 `torch==2.7.1+cpu`。
- CUDA 12.6 固定安装对应的 `torch==2.7.1+cu126`。
- 官方索引下载使用 180 秒超时、10 次重试和无缓存模式。
- 标准 TLS 下载失败后自动重试，并仅对当前命令中的 `download.pytorch.org` 放宽主机验证。
- 两轮下载都失败时给出切换网络或先采用 API 模式的明确处理建议。
- 学员仍然只需运行 `setup.ps1` 并选择 CPU、CUDA 或在线 API。
