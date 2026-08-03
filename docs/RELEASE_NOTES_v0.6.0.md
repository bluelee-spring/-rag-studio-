# v0.6.0：本地模型与工作区请求链修复

## 本次交付解决的问题

- 顶部“实验台 / 数据工作区 / 模型连接”增加当前页与跳转中的视觉状态，三个页面均可直接访问。
- 补齐 Next.js 缺失的模型测试、文档上传、表上传和任务轮询代理路由。
- 模型主路径统一识别完整基础模型、LoRA 已合并模型和 PEFT LoRA 输出。
- 直接填写 `outputs\run_xxx` 时，读取 `adapter_config.json` 并自动定位 `backend\models` 中的基础模型。
- 本地模型测试改为真实加载权重并完成短生成，测试通过后模型保留在 FastAPI 进程中。
- RAG 生成失败时把异常类型和消息写入生成阶段，并明确回退到确定性模板。
- `setup.ps1` 支持在线 API、本地 CPU、本地 CUDA 三种安装模式；学员不再逐条声明虚拟环境 Python。
- 新增 `check-environment.ps1`，统一检查 FastAPI、检索依赖、Node.js 和可选本地模型环境。

## 已执行验证

```text
后端 pytest：14 passed
前端 TypeScript：通过
Next.js 生产构建：通过
GET /、/settings、/data：200
POST /api/model/test：代理命中
文档上传 → 建库 → 轮询：202 → completed 100%
CSV 上传 → SQLite 建库 → 轮询：202 → completed 100%
```

模型权重本身不包含在交付包中。Windows 上的真实 CUDA 生成需要学员机器安装完成 CUDA 版 PyTorch，并由“保存并加载测试”在该机器上最终确认。
