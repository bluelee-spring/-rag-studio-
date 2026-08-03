# RAG Studio 批量分类功能改造指令

> **给 codewhale 的使用说明**：请按照本文档的改造清单，在当前项目目录下逐一修改/新建文件。每个代码块标注了目标文件路径，直接写入对应文件即可。不要修改文档中标注为"不需要修改"的文件。

## 目标

在现有 RAG Studio 项目中新增「批量分类」功能：用户上传 CSV 数据文件 → 前端选择查询列和分类配置 → 后端逐行执行 RAG 检索 + LLM 生成 → 返回分类结果并在前端表格展示。

**使用场景**：舆情评论三分类（相关·支持 / 相关·质疑 / 无关·噪音）。用户将含评论数据的 CSV 上传后，选择评论所在列，系统逐行将评论作为查询送入 RAG 检索流水线获取证据，再用自定义 prompt 调 LLM 生成分类结果。

---

## 项目技术栈

- 前端：Next.js 14 App Router + TypeScript（无 UI 框架，纯 CSS，样式在 `frontend/app/globals.css`）
- 后端：FastAPI + Pydantic（Python 3.11，无 ORM，直接 sqlite3）
- 前后端通信：前端 BFF 代理层（`frontend/app/api/*/route.ts`）→ 后端 FastAPI（`backend/app/main.py`）
- 现有查询流程：`POST /api/v1/rag/query` → `RagOrchestrator.run()` 或 `WorkspaceRagRouter.run()` → `AnswerGenerator.generate()`

---

## 改造清单

### 一、后端：新增批量分类端点

#### 1.1 新增数据模型 — `backend/app/models.py`

在文件末尾追加以下 Pydantic 模型（保持与现有模型的风格一致）：

```python
class BatchClassifyItem(BaseModel):
    row_index: int
    raw_data: dict[str, str]
    query_text: str
    classification: str = ""
    reasoning: str = ""
    evidence_snippet: str = ""
    status: Literal["pending", "success", "error"] = "pending"
    error: str = ""


class BatchClassifyRequest(BaseModel):
    csv_content: str = Field(min_length=10, max_length=5_000_000)
    query_column: str = Field(min_length=1, max_length=100)
    extra_columns: list[str] = Field(default_factory=list)
    mode: RagMode = "semantic"
    top_k: int = Field(default=5, ge=1, le=20)
    workspace_id: str = Field(
        default="builtin-soybean",
        min_length=3, max_length=80,
        pattern=r"^[a-z0-9][a-z0-9-]*$",
    )
    classification_prompt: str = Field(
        default=(
            "你是舆情评论分类助手。根据检索到的相关文档和上下文，"
            "判断以下评论属于哪个类别，只输出类别名称。\n"
            "类别选项：相关·支持 / 相关·质疑 / 无关·噪音\n"
            "评论内容：{query}\n"
            "检索证据：{evidence}\n"
            "请输出：类别名称|一句话理由"
        ),
        max_length=5000,
    )
    max_rows: int = Field(default=500, ge=1, le=5000)


class BatchClassifyResponse(BaseModel):
    total_rows: int
    processed: int
    succeeded: int
    failed: int
    items: list[BatchClassifyItem]
    columns: list[str]
    execution_source: str = "embedded"
```

#### 1.2 新增批量分类服务 — `backend/app/services/batch_classify.py`（新文件）

**关键设计决策**：不调用 `AnswerGenerator.generate()`，因为它的 prompt 是写死的大豆病害专用模板（"你是教学RAG系统的答案生成器"）。批量分类服务直接调用 `model_runtime.generate()` 使用自定义 prompt，同时复用现有的 `RagOrchestrator` / `WorkspaceRagRouter` 做检索。

```python
from __future__ import annotations

import csv
import io
import json

from ..models import (
    BatchClassifyItem,
    BatchClassifyRequest,
    BatchClassifyResponse,
    EvidenceItem,
    QueryRequest,
    TraceStage,
)
from .model_runtime import model_runtime
from .orchestrator import RagOrchestrator
from .workspace_query import WorkspaceRagRouter


class BatchClassifyService:
    """逐行读取 CSV，将指定列作为 query 传入 RAG 检索，再用自定义 prompt 调 LLM 生成分类。"""

    def __init__(
        self,
        orchestrator: RagOrchestrator,
        workspace_router: WorkspaceRagRouter,
    ) -> None:
        self._orchestrator = orchestrator
        self._workspace_router = workspace_router

    def run(self, request: BatchClassifyRequest) -> BatchClassifyResponse:
        # 1. 解析 CSV
        reader = csv.DictReader(io.StringIO(request.csv_content))
        if not reader.fieldnames:
            raise ValueError("CSV 文件没有列头")
        columns = list(reader.fieldnames)
        if request.query_column not in columns:
            raise ValueError(
                f"查询列 '{request.query_column}' 不存在于 CSV 列头中。"
                f"可用列：{columns}"
            )

        rows = list(reader)
        if len(rows) > request.max_rows:
            rows = rows[: request.max_rows]

        items: list[BatchClassifyItem] = []
        succeeded = 0
        failed = 0

        for idx, row in enumerate(rows):
            query_text = (row.get(request.query_column) or "").strip()
            if not query_text:
                items.append(BatchClassifyItem(
                    row_index=idx,
                    raw_data={k: str(v) for k, v in row.items()},
                    query_text="",
                    status="error",
                    error="查询列为空",
                ))
                failed += 1
                continue

            item = BatchClassifyItem(
                row_index=idx,
                raw_data={k: str(v) for k, v in row.items()},
                query_text=query_text,
            )

            try:
                # 2. 调用现有 RAG 检索流水线（复用 orchestrator / workspace_router）
                qr = QueryRequest(
                    mode=request.mode,
                    question=query_text[:2000],
                    top_k=request.top_k,
                    workspace_id=request.workspace_id,
                )
                if request.workspace_id != "builtin-soybean":
                    resp = self._workspace_router.run(qr)
                else:
                    resp = self._orchestrator.run(qr)

                # 3. 提取检索证据摘要
                evidence_text = ""
                if resp.evidence:
                    evidence_text = " | ".join(
                        f"[{e.title}] {e.excerpt[:200]}" for e in resp.evidence[:3]
                    )
                item.evidence_snippet = evidence_text[:500]

                # 4. 用自定义 prompt 直接调 LLM 生成分类
                #    （绕过 AnswerGenerator，因为它的大豆病害 prompt 不适用于分类场景）
                if model_runtime.answer_enabled:
                    user_prompt = request.classification_prompt.replace(
                        "{query}", query_text[:2000]
                    ).replace(
                        "{evidence}", evidence_text[:2000]
                    )
                    answer, provider, _ = model_runtime.generate(
                        system="你是一个文本分类助手，请严格按照指定格式输出分类结果。",
                        prompt=user_prompt,
                        temperature=0,
                    )
                    item.classification = _extract_label(answer)
                    item.reasoning = _extract_reasoning(answer)
                else:
                    # LLM 未配置时，用检索证据做简单匹配 fallback
                    item.classification = "未知（LLM未配置）"
                    item.reasoning = f"检索到 {len(resp.evidence)} 条证据，但 LLM 未启用"
                item.status = "success"
                succeeded += 1

            except Exception as exc:
                item.status = "error"
                item.error = str(exc)[:300]
                failed += 1

            items.append(item)

        return BatchClassifyResponse(
            total_rows=len(rows),
            processed=len(items),
            succeeded=succeeded,
            failed=failed,
            items=items,
            columns=columns,
        )


def _extract_label(answer: str) -> str:
    """从 LLM 回答中提取类别标签。"""
    for label in ["相关·支持", "相关·质疑", "无关·噪音"]:
        if label in answer:
            return label
    # fallback: 取第一行
    first_line = answer.strip().split("\n")[0]
    return first_line[:50] if first_line else "未知"


def _extract_reasoning(answer: str) -> str:
    """从 LLM 回答中提取理由。"""
    if "|" in answer:
        parts = answer.split("|", 1)
        if len(parts) > 1:
            return parts[1].strip()[:300]
    lines = answer.strip().split("\n")
    if len(lines) > 1:
        return "\n".join(lines[1:]).strip()[:300]
    return answer.strip()[:300]
```

#### 1.3 新增 API 端点 — `backend/app/main.py`

在文件顶部的 import 区域添加：

```python
from .models import (
    # ... 已有的 imports ...
    BatchClassifyRequest,
    BatchClassifyResponse,
)
from .services.batch_classify import BatchClassifyService
```

在 `orchestrator` / `workspace_router` 初始化之后添加：

```python
batch_classify_service = BatchClassifyService(orchestrator, workspace_router)
```

在文件末尾（`run_query` 端点之后）添加新端点：

```python
@app.post("/api/v1/batch/classify", response_model=BatchClassifyResponse)
def run_batch_classify(request: BatchClassifyRequest) -> BatchClassifyResponse:
    try:
        return batch_classify_service.run(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"批量分类执行失败：{exc}",
        ) from exc
```

---

### 二、前端 BFF 代理：新增批量分类路由

#### 2.1 新建 `frontend/app/api/batch/route.ts`

**注意**：必须与现有 `frontend/app/api/rag/route.ts` 的写法保持一致，用直接 fetch + `process.env.RAG_BACKEND_URL`，不要用 `backend-proxy.ts` 里的辅助函数（现有路由也没用）。

```typescript
import { NextRequest, NextResponse } from "next/server";

const backend =
  process.env.RAG_BACKEND_URL ?? "http://127.0.0.1:8000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const upstream = await fetch(`${backend}/api/v1/batch/classify`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      cache: "no-store",
      // 批量处理可能耗时较长，设 600 秒超时
      signal: AbortSignal.timeout(600_000),
    });

    const payload = await upstream.json();
    return NextResponse.json(payload, {
      status: upstream.status,
    });
  } catch (error) {
    const detail =
      error instanceof Error && error.name === "TimeoutError"
        ? "后端批量执行超过600秒"
        : "无法连接FastAPI后端";
    return NextResponse.json({ detail }, { status: 502 });
  }
}
```

---

### 三、前端类型定义

#### 3.1 修改 `frontend/lib/types.ts`

在文件末尾追加：

```typescript
export type BatchClassifyItem = {
  row_index: number;
  raw_data: Record<string, string>;
  query_text: string;
  classification: string;
  reasoning: string;
  evidence_snippet: string;
  status: "pending" | "success" | "error";
  error: string;
};

export type BatchClassifyRequest = {
  csv_content: string;
  query_column: string;
  extra_columns?: string[];
  mode: RagMode;
  top_k: number;
  workspace_id: string;
  classification_prompt?: string;
  max_rows?: number;
};

export type BatchClassifyResponse = {
  total_rows: number;
  processed: number;
  succeeded: number;
  failed: number;
  items: BatchClassifyItem[];
  columns: string[];
  execution_source: string;
};
```

---

### 四、前端 API 函数

#### 4.1 修改 `frontend/lib/api.ts`

添加导入和函数：

```typescript
// 在文件顶部 import 中添加 BatchClassifyRequest, BatchClassifyResponse
import type {
  // ... 已有的 ...
  BatchClassifyRequest,
  BatchClassifyResponse,
} from "./types";

// 在文件末尾添加
export async function runBatchClassify(
  input: BatchClassifyRequest,
): Promise<BatchClassifyResponse> {
  const response = await fetch("/api/batch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "批量分类执行失败");
  }
  return payload as BatchClassifyResponse;
}
```

---

### 五、前端页面：批量分类工作台

#### 5.1 新建 `frontend/app/batch/page.tsx`

这是一个完整的页面组件，包含 4 个阶段：

**阶段 1 — 上传 CSV**：文件选择 → 读取内容 → 解析列头
**阶段 2 — 配置**：选择查询列、选择 RAG 模式、选择工作区、编辑分类 Prompt
**阶段 3 — 预览**：显示 CSV 前 10 行预览表格
**阶段 4 — 执行 + 结果**：点击运行 → 显示进度 → 结果表格 → 下载按钮

```tsx
"use client";

import { useState, useCallback, useRef } from "react";
import { PlatformHeader } from "@/components/PlatformHeader";
import {
  loadConfig,
  runBatchClassify,
} from "@/lib/api";
import type {
  BatchClassifyResponse,
  ModeInfo,
  WorkspaceInfo,
} from "@/lib/types";
import { useEffect } from "react";

const DEFAULT_PROMPT = `你是舆情评论分类助手。根据检索到的相关文档和上下文，判断以下评论属于哪个类别，只输出类别名称。
类别选项：相关·支持 / 相关·质疑 / 无关·噪音
评论内容：{query}
检索证据：{evidence}
请输出：类别名称|一句话理由`;

export default function BatchPage() {
  // --- 状态 ---
  const [csvContent, setCsvContent] = useState("");
  const [csvName, setCsvName] = useState("");
  const [columns, setColumns] = useState<string[]>([]);
  const [previewRows, setPreviewRows] = useState<Record<string, string>[]>([]);
  const [queryColumn, setQueryColumn] = useState("");
  const [mode, setMode] = useState<string>("semantic");
  const [workspaceId, setWorkspaceId] = useState("builtin-soybean");
  const [topK, setTopK] = useState(5);
  const [maxRows, setMaxRows] = useState(500);
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [modes, setModes] = useState<ModeInfo[]>([]);
  const [workspaces, setWorkspaces] = useState<WorkspaceInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BatchClassifyResponse | null>(null);
  const [error, setError] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- 加载配置 ---
  useEffect(() => {
    loadConfig()
      .then((cfg) => {
        setModes(cfg.modes);
        setWorkspaces(cfg.workspaces);
      })
      .catch(() => {});
  }, []);

  // --- CSV 文件处理 ---
  const handleFileUpload = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      setError("");
      setResult(null);
      const reader = new FileReader();
      reader.onload = () => {
        const text = String(reader.result || "");
        setCsvContent(text);
        setCsvName(file.name);
        // 解析列头和前 10 行预览
        const lines = text.split(/\r?\n/).filter((l) => l.trim());
        if (lines.length === 0) {
          setError("CSV 文件为空");
          return;
        }
        // 简单 CSV 解析（不处理引号内逗号的边缘情况也可以接受）
        const parseLine = (line: string): string[] => {
          // 处理带引号的 CSV
          const result: string[] = [];
          let current = "";
          let inQuotes = false;
          for (let i = 0; i < line.length; i++) {
            const ch = line[i];
            if (ch === '"') {
              if (inQuotes && line[i + 1] === '"') {
                current += '"';
                i++;
              } else {
                inQuotes = !inQuotes;
              }
            } else if (ch === "," && !inQuotes) {
              result.push(current);
              current = "";
            } else {
              current += ch;
            }
          }
          result.push(current);
          return result;
        };
        const headers = parseLine(lines[0]);
        setColumns(headers);
        if (headers.length > 0) {
          setQueryColumn(headers[0]);
        }
        const rows: Record<string, string>[] = [];
        for (let i = 1; i < Math.min(lines.length, 11); i++) {
          const values = parseLine(lines[i]);
          const row: Record<string, string> = {};
          headers.forEach((h, j) => {
            row[h] = values[j] || "";
          });
          rows.push(row);
        }
        setPreviewRows(rows);
      };
      reader.readAsText(file, "utf-8");
    },
    [],
  );

  // --- 执行批量分类 ---
  const handleRun = useCallback(async () => {
    if (!csvContent || !queryColumn) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const resp = await runBatchClassify({
        csv_content: csvContent,
        query_column: queryColumn,
        mode: mode as any,
        top_k: topK,
        workspace_id: workspaceId,
        classification_prompt: prompt,
        max_rows: maxRows,
      });
      setResult(resp);
    } catch (e: any) {
      setError(e.message || "批量分类失败");
    } finally {
      setLoading(false);
    }
  }, [csvContent, queryColumn, mode, topK, workspaceId, prompt, maxRows]);

  // --- 下载结果 ---
  const handleDownload = useCallback(() => {
    if (!result) return;
    const headers = [...result.columns, "分类结果", "理由", "状态"];
    const lines = [headers.join(",")];
    for (const item of result.items) {
      const rowValues = result.columns.map(
        (c) => `"${(item.raw_data[c] || "").replace(/"/g, '""')}"`,
      );
      rowValues.push(`"${item.classification.replace(/"/g, '""')}"`);
      rowValues.push(`"${item.reasoning.replace(/"/g, '""')}"`);
      rowValues.push(item.status);
      lines.push(rowValues.join(","));
    }
    const blob = new Blob(["\ufeff" + lines.join("\n")], {
      type: "text/csv;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `batch_result_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [result]);

  return (
    <>
      <PlatformHeader />
      <main className="batch-page" style={{ padding: "24px 32px", maxWidth: 1400, margin: "0 auto" }}>
        <h1 style={{ fontSize: 24, marginBottom: 4 }}>批量分类工作台</h1>
        <p style={{ color: "#666", marginBottom: 24, fontSize: 14 }}>
          上传 CSV 数据文件，选择查询列，系统将逐行调用 RAG 检索 + LLM 生成分类结果。
        </p>

        {/* 阶段 1: 上传 */}
        <section style={{ marginBottom: 32 }}>
          <h2 style={{ fontSize: 18, marginBottom: 12 }}>1. 上传 CSV 文件</h2>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            onChange={handleFileUpload}
            style={{ display: "none" }}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            style={{
              padding: "10px 24px",
              border: "2px dashed #ccc",
              borderRadius: 8,
              cursor: "pointer",
              background: "#fafafa",
            }}
          >
            {csvName ? `已选择: ${csvName}` : "点击选择 CSV 文件"}
          </button>
        </section>

        {/* 阶段 2: 配置 */}
        {columns.length > 0 && (
          <section style={{ marginBottom: 32 }}>
            <h2 style={{ fontSize: 18, marginBottom: 12 }}>2. 配置分类参数</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 13, fontWeight: 600 }}>查询列（每行该列内容作为查询）</span>
                <select
                  value={queryColumn}
                  onChange={(e) => setQueryColumn(e.target.value)}
                  style={{ padding: "8px 12px", borderRadius: 6, border: "1px solid #ddd" }}
                >
                  {columns.map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 13, fontWeight: 600 }}>RAG 检索模式</span>
                <select
                  value={mode}
                  onChange={(e) => setMode(e.target.value)}
                  style={{ padding: "8px 12px", borderRadius: 6, border: "1px solid #ddd" }}
                >
                  {modes.map((m) => (
                    <option key={m.id} value={m.id}>{m.name} — {m.summary}</option>
                  ))}
                </select>
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 13, fontWeight: 600 }}>工作区</span>
                <select
                  value={workspaceId}
                  onChange={(e) => setWorkspaceId(e.target.value)}
                  style={{ padding: "8px 12px", borderRadius: 6, border: "1px solid #ddd" }}
                >
                  {workspaces.map((w) => (
                    <option key={w.id} value={w.id}>{w.name} ({w.kind})</option>
                  ))}
                </select>
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 13, fontWeight: 600 }}>Top-K 检索数量</span>
                <input
                  type="number"
                  min={1}
                  max={20}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  style={{ padding: "8px 12px", borderRadius: 6, border: "1px solid #ddd" }}
                />
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 13, fontWeight: 600 }}>最大处理行数</span>
                <input
                  type="number"
                  min={1}
                  max={5000}
                  value={maxRows}
                  onChange={(e) => setMaxRows(Number(e.target.value))}
                  style={{ padding: "8px 12px", borderRadius: 6, border: "1px solid #ddd" }}
                />
              </label>
            </div>
            <label style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 16 }}>
              <span style={{ fontSize: 13, fontWeight: 600 }}>分类 Prompt 模板</span>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={6}
                style={{
                  padding: "12px",
                  borderRadius: 6,
                  border: "1px solid #ddd",
                  fontFamily: "monospace",
                  fontSize: 13,
                }}
              />
              <span style={{ fontSize: 12, color: "#999" }}>
                {"{query} 会被替换为每行的查询文本，{evidence} 会被替换为检索到的证据片段"}
              </span>
            </label>
          </section>
        )}

        {/* 阶段 3: 预览 */}
        {previewRows.length > 0 && (
          <section style={{ marginBottom: 32 }}>
            <h2 style={{ fontSize: 18, marginBottom: 12 }}>3. 数据预览（前 10 行）</h2>
            <div style={{ overflowX: "auto", border: "1px solid #eee", borderRadius: 8 }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr style={{ background: "#f5f5f5" }}>
                    <th style={{ padding: "8px 12px", textAlign: "left", borderBottom: "1px solid #ddd" }}>#</th>
                    {columns.map((c) => (
                      <th
                        key={c}
                        style={{
                          padding: "8px 12px",
                          textAlign: "left",
                          borderBottom: "1px solid #ddd",
                          color: c === queryColumn ? "#2F6B57" : "inherit",
                          fontWeight: c === queryColumn ? 700 : 600,
                        }}
                      >
                        {c}
                        {c === queryColumn ? " ← 查询列" : ""}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {previewRows.map((row, i) => (
                    <tr key={i}>
                      <td style={{ padding: "8px 12px", color: "#999" }}>{i + 1}</td>
                      {columns.map((c) => (
                        <td
                          key={c}
                          style={{
                            padding: "8px 12px",
                            borderBottom: "1px solid #f0f0f0",
                            maxWidth: 300,
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                          }}
                        >
                          {row[c]}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* 阶段 4: 执行 + 结果 */}
        {columns.length > 0 && (
          <section style={{ marginBottom: 32 }}>
            <h2 style={{ fontSize: 18, marginBottom: 12 }}>4. 执行批量分类</h2>
            <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
              <button
                onClick={handleRun}
                disabled={loading || !csvContent || !queryColumn}
                style={{
                  padding: "10px 32px",
                  borderRadius: 8,
                  border: "none",
                  background: loading ? "#ccc" : "#2F6B57",
                  color: "#fff",
                  cursor: loading ? "not-allowed" : "pointer",
                  fontSize: 15,
                  fontWeight: 600,
                }}
              >
                {loading ? "正在处理..." : "开始批量分类"}
              </button>
              {result && (
                <button
                  onClick={handleDownload}
                  style={{
                    padding: "10px 24px",
                    borderRadius: 8,
                    border: "1px solid #2F6B57",
                    background: "#fff",
                    color: "#2F6B57",
                    cursor: "pointer",
                    fontSize: 15,
                  }}
                >
                  下载结果 CSV
                </button>
              )}
            </div>

            {error && (
              <div style={{
                padding: "12px 16px",
                background: "#fee",
                border: "1px solid #fcc",
                borderRadius: 8,
                color: "#c33",
                marginBottom: 16,
              }}>
                {error}
              </div>
            )}

            {result && (
              <div>
                {/* 统计摘要 */}
                <div style={{ display: "flex", gap: 24, marginBottom: 16 }}>
                  <div style={{
                    padding: "12px 20px",
                    background: "#f0f7f4",
                    borderRadius: 8,
                    border: "1px solid #d0e8de",
                  }}>
                    <div style={{ fontSize: 12, color: "#666" }}>总行数</div>
                    <div style={{ fontSize: 24, fontWeight: 700 }}>{result.total_rows}</div>
                  </div>
                  <div style={{
                    padding: "12px 20px",
                    background: "#f0f7f4",
                    borderRadius: 8,
                    border: "1px solid #d0e8de",
                  }}>
                    <div style={{ fontSize: 12, color: "#666" }}>成功</div>
                    <div style={{ fontSize: 24, fontWeight: 700, color: "#2F6B57" }}>{result.succeeded}</div>
                  </div>
                  <div style={{
                    padding: "12px 20px",
                    background: "#fef0f0",
                    borderRadius: 8,
                    border: "1px solid #fcd0d0",
                  }}>
                    <div style={{ fontSize: 12, color: "#666" }}>失败</div>
                    <div style={{ fontSize: 24, fontWeight: 700, color: "#c33" }}>{result.failed}</div>
                  </div>
                </div>

                {/* 结果表格 */}
                <div style={{ overflowX: "auto", border: "1px solid #eee", borderRadius: 8 }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                    <thead>
                      <tr style={{ background: "#f5f5f5" }}>
                        <th style={{ padding: "8px 12px", textAlign: "left", borderBottom: "1px solid #ddd" }}>#</th>
                        <th style={{ padding: "8px 12px", textAlign: "left", borderBottom: "1px solid #ddd" }}>查询内容</th>
                        <th style={{ padding: "8px 12px", textAlign: "left", borderBottom: "1px solid #ddd" }}>分类结果</th>
                        <th style={{ padding: "8px 12px", textAlign: "left", borderBottom: "1px solid #ddd" }}>理由</th>
                        <th style={{ padding: "8px 12px", textAlign: "left", borderBottom: "1px solid #ddd" }}>状态</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.items.map((item) => (
                        <tr key={item.row_index}>
                          <td style={{ padding: "8px 12px", color: "#999" }}>{item.row_index + 1}</td>
                          <td style={{
                            padding: "8px 12px",
                            maxWidth: 300,
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                          }}>
                            {item.query_text}
                          </td>
                          <td style={{ padding: "8px 12px", fontWeight: 600 }}>
                            <span style={{
                              display: "inline-block",
                              padding: "2px 8px",
                              borderRadius: 4,
                              fontSize: 12,
                              background: item.classification.includes("支持") ? "#e8f5e9" :
                                          item.classification.includes("质疑") ? "#fff3e0" :
                                          item.classification.includes("噪音") ? "#fce4ec" : "#f0f0f0",
                              color: item.classification.includes("支持") ? "#2e7d32" :
                                     item.classification.includes("质疑") ? "#e65100" :
                                     item.classification.includes("噪音") ? "#c62828" : "#666",
                            }}>
                              {item.classification || "—"}
                            </span>
                          </td>
                          <td style={{
                            padding: "8px 12px",
                            maxWidth: 400,
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                            color: "#666",
                          }}>
                            {item.reasoning || item.error || "—"}
                          </td>
                          <td style={{ padding: "8px 12px" }}>
                            <span style={{
                              color: item.status === "success" ? "#2e7d32" :
                                     item.status === "error" ? "#c62828" : "#999",
                            }}>
                              {item.status === "success" ? "✓" : item.status === "error" ? "✗" : "…"}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </section>
        )}
      </main>
    </>
  );
}
```

---

### 六、导航入口

#### 6.1 修改 `frontend/components/PlatformHeader.tsx`

在 `NAVIGATION` 数组中添加新条目：

```typescript
const NAVIGATION = [
  { href: "/", label: "实验台" },
  { href: "/data", label: "数据工作区" },
  { href: "/batch", label: "批量分类" },    // ← 新增
  { href: "/settings", label: "模型连接" },
];
```

---

## 改造总结

| 改动文件 | 类型 | 说明 |
|----------|------|------|
| `backend/app/models.py` | 修改 | 新增 3 个 Pydantic 模型（BatchClassifyItem / Request / Response） |
| `backend/app/services/batch_classify.py` | **新建** | 批量分类服务：解析 CSV → 逐行调用 RAG 检索 → 直接调 model_runtime.generate() 用自定义 prompt 生成分类 |
| `backend/app/main.py` | 修改 | 新增 import + 初始化 service + 1 个 `POST /api/v1/batch/classify` 端点 |
| `frontend/app/api/batch/route.ts` | **新建** | BFF 代理路由（600 秒超时，与 rag/route.ts 风格一致） |
| `frontend/lib/types.ts` | 修改 | 新增 3 个 TypeScript 类型 |
| `frontend/lib/api.ts` | 修改 | 新增 `runBatchClassify()` 函数 |
| `frontend/app/batch/page.tsx` | **新建** | 批量分类工作台页面（上传→配置→预览→执行→结果→下载） |
| `frontend/components/PlatformHeader.tsx` | 修改 | 导航栏 NAVIGATION 数组新增 `{ href: "/batch", label: "批量分类" }` |

**不需要修改的文件**：`answer_generator.py`、`orchestrator.py`、`workspace_query.py`、`model_runtime.py`、`workspaces.py`——批量分类服务直接复用它们的现有接口。

1. **复用现有检索流水线，但绕过 AnswerGenerator**：`BatchClassifyService` 内部调用 `orchestrator.run(QueryRequest)` 或 `workspace_router.run(QueryRequest)` 获取检索证据（`resp.evidence`），但**不使用 `resp.answer`**——因为 `AnswerGenerator` 的 prompt 是写死的大豆病害专用模板（"你是教学RAG系统的答案生成器。只能基于输入证据作答，不得改写数值"），不适用于舆情分类场景。批量分类服务直接调用 `model_runtime.generate(system=..., prompt=...)` 使用用户自定义的分类 prompt。
2. **CSV 编码**：前端用 `FileReader.readAsText(file, "utf-8")` 读取。如果 CSV 是 GBK/GB2312 编码会乱码——后端 `workspaces.py` 的 `_read_csv()` 已有编码检测逻辑（utf-8-sig → utf-8 → gb18030），可以参考。建议在前端加一个编码选择下拉框（UTF-8 / GBK / 自动检测），或者直接在后端做编码检测。
3. **超时**：批量处理 500 行可能需要几分钟，前端 BFF 超时设为 600 秒，后端 FastAPI 同步端点不设超时。
4. **不做分页**：结果表格一次性渲染所有行，max_rows 已限制为 5000。如果性能不够可以后续加分页。
5. **model_runtime.generate() 签名**：该方法返回 `(answer: str, provider: str, extra: dict)` 三元组，temperature 参数控制随机性。批量分类用 `temperature=0` 保证结果可复现。
6. **LLM 未配置时的 fallback**：如果 `model_runtime.answer_enabled` 为 False（用户没在 /settings 页面配置 API），批量分类仍然会执行检索，只是分类结果会标记为"未知（LLM未配置）"。前端应该检测 model 状态并提示用户先配置模型。

---

## 验证清单（改完后逐项检查）

- [ ] 后端启动无报错，`POST /api/v1/batch/classify` 端点可访问
- [ ] 前端 `npm run build` 无 TypeScript 错误
- [ ] 导航栏出现「批量分类」入口，点击可进入 `/batch` 页面
- [ ] 上传 CSV 后列头正确解析，下拉框可选择查询列
- [ ] 数据预览表格正确显示前 10 行
- [ ] 点击「开始批量分类」后，后端控制台有逐行处理日志
- [ ] 结果表格显示分类结果、理由、状态
- [ ] 「下载结果 CSV」按钮可下载带分类结果的 CSV 文件
- [ ] LLM 未配置时，分类结果显示"未知（LLM未配置）"而非报错
