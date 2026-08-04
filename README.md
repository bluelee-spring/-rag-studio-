# RAG Studio — 检索增强生成方法实验台

> RAG Teaching Studio v0.7.0 · 多范式检索可视化 · 批量分类工作台

**RAG Studio** 是一个面向教学与实验的检索增强生成（Retrieval-Augmented Generation）可视化平台。它在单一界面内集成了 **七种检索范式**——从经典的 TF–IDF、BM25 到语义向量、属性图、RDF/SPARQL、关系数据库 SQL，以及多检索器综合路由——让每种方法的内部机制变得可见、可审计、可对比。

---

## 项目截图

| 实验台主界面 | 数据工作区 | 批量分类结果 |
|:---:|:---:|:---:|
| ![实验台](./docs/屏幕截图%202026-08-03%20173430.png) | ![数据工作区](./docs/屏幕截图%202026-08-03%20211835.png) | ![批量分类](./docs/屏幕截图%202026-08-04%20161603.png) |

---

## 核心特性

### 🔍 七种检索范式，同台对比

| 模式 | 检索方法 | 核心可视化 |
|------|----------|------------|
| **TF–IDF** | 词频 × 逆文档频率 + 余弦相似度 | 110 维词表空间点亮、逐维乘积、余弦分子/分母分解 |
| **BM25** | 词频饱和 + 长度归一化 + 稀疏排名 | DF/IDF 表、词项贡献矩阵、饱和度曲线、分数求和总线 |
| **语义向量** | Embedding + FAISS 精确内积搜索 | 192 维稠密向量、FAISS 索引构建、查询探针、Top-K 回收 |
| **属性图** | 实体锚点 + 受约束 Cypher 遍历 | 三维语义空间锚点选择、关系白名单、逐跳子图扩展 |
| **RDF / SPARQL** | IRI 本体推理 + 三元组匹配 | Class/Property/IRI 映射、变量槽绑定、安全间隔期闸门 |
| **关系数据库** | 参数化 SQL + 查询计划 | 物理行过滤、等值键连接、GROUP BY 聚合桶 |
| **综合路由** | 多检索器任务分解 + 证据合并 | 问题拆解 → 各检索器独立执行 → 证据融合 → LLM 生成 |

### 📊 批量分类工作台

新增的批量分类功能支持：
- 上传任意 CSV，选择查询列
- 逐行调用 RAG 检索 + 自定义 Prompt + LLM 生成分类
- 三分类标签（舆情评论：相关·支持 / 相关·质疑 / 无关·噪音）
- 结果表格预览 + 一键下载带分类结果的 CSV

### 🎓 教学友好的设计

- **每轮查询可逐帧复盘**：点击阶段名称暂停，教师可逐项讲解算法内部状态
- **所有数值来自后端执行轨迹**：前端只负责投影，不伪造检索分数
- **LLM 可回退**：模型未配置时自动降级为确定性模板，展示 RAG 的检索层和生成层可独立失败
- **内置教学数据**：228 个文本块、大豆病害知识图谱、RDF 本体、SQLite 病例库，开箱即用

---

## 技术架构

```
Next.js 14 (App Router)          ← 前端实验台 + BFF 代理
        │
FastAPI (Python 3.11)            ← 检索执行器 + 答案生成
        │
   ┌────┼────────────┬──────────────┐
   ▼    ▼            ▼              ▼
 FAISS  Jieba     RDFLib        SQLite
(向量) (分词)    (RDF推理)    (关系查询)
```

- **前端**：Next.js 14 + TypeScript，Canvas 透视投影 + 深度排序可视化
- **后端**：FastAPI，计划器-执行器分离架构
- **文档检索**：Jieba 分词 + TF–IDF / BM25 + FAISS IndexFlatIP
- **属性图**：SQLite 邻接索引 + FAISS 实体锚点，不依赖 Neo4j 也可离线运行
- **RDF**：RDFLib 内存图（支持 Fuseki 可选接入）
- **SQL**：SQLite `query_only` + VM authorizer，LLM 只提候选参数，不能直接写数据库

---

## 快速开始

### 环境要求

- Python 3.10 – 3.12
- Node.js ≥ 18
- Windows / macOS / Linux

### 一键安装

```powershell
# 在线 API 模式（最快，不装 PyTorch）
.\setup.ps1
# 按提示选 [1]

# 或本地模型 CPU / CUDA
.\setup.ps1 -Mode cpu
```

### 启动

```powershell
.\start.ps1
```

然后访问：
- 实验台：http://localhost:3000
- 数据工作区：http://localhost:3000/data
- 批量分类：http://localhost:3000/batch
- 模型配置：http://localhost:3000/settings

### 配置 LLM（可选）

在「模型连接」页面填入 OpenAI 兼容的 API 信息（如 DeepSeek），或加载本地 Hugging Face 模型。不配置也能运行——系统会自动回退到确定性模板。

---

## 项目结构

```
rag-teaching-studio/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI 入口
│   │   ├── config.py            # 配置
│   │   ├── models.py            # Pydantic 数据模型
│   │   └── services/
│   │       ├── document_rag.py  # TF–IDF / BM25 / 语义向量
│   │       ├── property_graph_rag.py  # 属性图检索
│   │       ├── rdf_rag.py       # RDF / SPARQL
│   │       ├── sql_rag.py       # 参数化 SQL
│   │       ├── orchestrator.py  # 综合路由编排
│   │       ├── batch_classify.py # 批量分类服务
│   │       └── model_runtime.py # LLM 运行时
├── frontend/
│   ├── app/
│   │   ├── page.tsx             # 实验台主页
│   │   ├── data/page.tsx        # 数据工作区
│   │   ├── batch/page.tsx       # 批量分类工作台
│   │   └── settings/page.tsx    # 模型连接
│   └── components/
│       ├── PipelinePlayer.tsx    # 检索管道播放器
│       ├── GraphView.tsx        # 图可视化
│       └── SpatialScene.tsx     # 三维向量空间
├── data/                        # 内置教学数据
│   ├── documents/               # 228 个文本块 + FAISS 索引
│   ├── graph/                   # 属性图 + 本体
│   └── relational/              # SQLite 病例库
├── runtime/                     # 用户上传数据（不提交 Git）
└── docs/                        # 文档
```

---

## 许可证

本项目仅用于教学和实验目的。内置全部病例、药剂与安全间隔期均为模拟数据，不可用于真实诊断或生产决策。

---

## 相关资源

- [系统架构文档](./docs/ARCHITECTURE.md)
- [授课建议](./docs/TEACHING_GUIDE.md)
- [RAG-Studio 运行讲义](./docs/RAG-Studio-Windows-macOS-运行讲义(1).md)
- [项目讲解](./docs/RAG-Teaching-Studio-项目讲解.md)
