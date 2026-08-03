# RAG Studio 批量分类页面 UI 重构指令

## 问题描述

新创建的 `frontend/app/batch/page.tsx` 使用了**大量 inline styles**（`style={{ padding: "24px 32px", ... }}`），完全绕开了项目原有的设计 token 系统，导致两个问题：

1. **页面顶部被遮挡**：缺少 `.platform-frame` 外层包裹，`.platform-page` 也没有正确的 `padding-top`，导致 PlatformHeader 把 H1 标题遮住了一半。
2. **样式与其它页面割裂**：用了大块灰色背景、粗黑边框，看起来像 2008 年的内部工具，不符合"数据工作区"/"模型连接"页面那种精致的实验室风格。

## 项目设计系统（必须复用）

所有页面统一遵循这个三层结构（参考 `frontend/app/data/page.tsx`）：

```tsx
<div className="platform-frame">
  <PlatformHeader serviceOnline={...} modelReady={...} />
  <main className="platform-page batch-page">    {/* batch-page 是页面专属 class */}
    <header className="platform-page-head">
      <div>
        <span>BATCH CLASSIFICATION</span>           {/* 副标题：9px 大写、letter-spacing 0.2em */}
        <h1>批量分类工作台</h1>                      {/* 主标题：Georgia 衬线体，34-53px */}
        <p>逐行调用 RAG 检索与 LLM，生成分类结果。</p>
      </div>
      <div className="workspace-count-card">      {/* 右上角深色卡片，显示状态 */}
        <span>ROWS</span>
        <strong>{String(totalRows).padStart(2, "0")}</strong>
        <small>待处理数据行</small>
      </div>
    </header>
    {/* 后续 sections */}
  </main>
</div>
```

**可用 class 速查**（来自 `frontend/app/globals.css`）：

| class | 用途 | 备注 |
|-------|------|------|
| `.platform-frame` | 最外层 | `min-height: 100vh` |
| `.platform-page` | main 容器 | 自动处理 `padding-top: calc(var(--topbar) + 58px)`，这是 PlatformHeader 不被遮挡的关键 |
| `.platform-page-head` | Hero header | flex 两端对齐 |
| `.workspace-count-card` / `.runtime-card` | 右上角深色卡片 | 复用现成的暗色样式 |
| `.drop-field` | 文件上传拖拽区 | 与数据工作区完全一致 |
| `.field` | 表单字段容器 | label > span + input/select |
| `.workspace-selector` | 下拉框容器 | 带下边框 |
| `.ingestion-monitor` | 进度监控卡 | 显示批量执行进度 |
| `.progress-rail` | 进度条轨道 | `<i style={{ width: `${progress}%` }} />` |
| `.ingestion-stages` + `.ingestion-stage` | 阶段列表 | 显示"已处理 23/100"等 |
| `.ingestion-result-strip` | 结果统计条 | 显示 total/success/failed 数字 |
| `.ingestion-warnings` | 警告区 | 显示失败项 |
| `.workspace-catalog` | 表格容器 | 可复用为结果表格容器 |
| `.open-workspace-button` | CTA 按钮 | 复用为主执行按钮 |
| `.build-button` | 构建按钮 | 也可复用 |

---

## 重构后的完整页面代码

**直接覆盖整个文件** `frontend/app/batch/page.tsx`：

```tsx
"use client";

import {
  ChangeEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { PlatformHeader } from "@/components/PlatformHeader";
import {
  loadConfig,
  runBatchClassify,
} from "@/lib/api";
import type {
  BatchClassifyItem,
  BatchClassifyResponse,
  ModeInfo,
  WorkspaceInfo,
} from "@/lib/types";

const DEFAULT_PROMPT = `你是舆情评论分类助手。根据检索到的相关文档和上下文，判断以下评论属于哪个类别，只输出类别名称。
类别选项：相关·支持 / 相关·质疑 / 无关·噪音
评论内容：{query}
检索证据：{evidence}
请输出：类别名称|一句话理由`;

function parseCsvLine(line: string): string[] {
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
}

export default function BatchPage() {
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
  const [serviceOnline, setServiceOnline] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<BatchClassifyResponse | null>(null);
  const [error, setError] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadConfig()
      .then((cfg) => {
        setModes(cfg.modes);
        setWorkspaces(cfg.workspaces);
        setServiceOnline(true);
        setModelReady(cfg.model.generation_ready);
      })
      .catch((err) => {
        setError(err.message);
        setServiceOnline(false);
      });
  }, []);

  const stats = useMemo(() => {
    if (!result) return { total: 0, succeeded: 0, failed: 0 };
    return {
      total: result.total_rows,
      succeeded: result.succeeded,
      failed: result.failed,
    };
  }, [result]);

  const handleFileUpload = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      setError("");
      setResult(null);
      const reader = new FileReader();
      reader.onload = () => {
        const text = String(reader.result || "");
        setCsvContent(text);
        setCsvName(file.name);
        const lines = text.split(/\r?\n/).filter((l) => l.trim());
        if (lines.length === 0) {
          setError("CSV 文件为空");
          return;
        }
        const headers = parseCsvLine(lines[0]);
        setColumns(headers);
        if (headers.length > 0) setQueryColumn(headers[0]);
        const rows: Record<string, string>[] = [];
        for (let i = 1; i < Math.min(lines.length, 11); i++) {
          const values = parseCsvLine(lines[i]);
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

  const handleRun = useCallback(async () => {
    if (!csvContent || !queryColumn) return;
    setLoading(true);
    setError("");
    setResult(null);
    setProgress(0);
    try {
      const resp = await runBatchClassify({
        csv_content: csvContent,
        query_column: queryColumn,
        mode: mode as never,
        top_k: topK,
        workspace_id: workspaceId,
        classification_prompt: prompt,
        max_rows: maxRows,
      });
      setResult(resp);
      setProgress(100);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "批量分类失败";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [csvContent, queryColumn, mode, topK, workspaceId, prompt, maxRows]);

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
    <div className="platform-frame">
      <PlatformHeader
        serviceOnline={serviceOnline}
        modelReady={modelReady}
      />
      <main className="platform-page batch-page">
        {/* ===== Hero header ===== */}
        <header className="platform-page-head">
          <div>
            <span>BATCH CLASSIFICATION</span>
            <h1>批量分类工作台</h1>
            <p>
              上传 CSV 数据文件，选择查询列，系统将逐行调用 RAG 检索与 LLM，生成分类结果。
            </p>
          </div>
          <div className="workspace-count-card">
            <span>ROWS</span>
            <strong>{String(stats.total || previewRows.length).padStart(2, "0")}</strong>
            <small>{result ? "已处理数据行" : "预览数据行"}</small>
          </div>
        </header>

        {/* ===== 1. 上传 CSV ===== */}
        <section className="batch-section batch-upload">
          <header>
            <div>
              <span>STEP 01</span>
              <h2>上传 CSV 数据文件</h2>
              <p>UTF-8 编码的 CSV，第一行为列头。文件不会上传到外部服务。</p>
            </div>
          </header>
          <div
            className={csvName ? "drop-field has-file" : "drop-field"}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click();
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              style={{ display: "none" }}
            />
            <div className="drop-volume" aria-hidden="true">
              <i />
              <i />
              <i />
              <b>CSV</b>
            </div>
            <strong>{csvName ? `已选择：${csvName}` : "点击选择 CSV 文件"}</strong>
            <span>支持 UTF-8 编码 · 第一行必须为列头 · 单文件上限 5 MB</span>
            <small>{previewRows.length > 0 ? `已解析 ${columns.length} 列，预览前 10 行` : ""}</small>
          </div>
        </section>

        {/* ===== 2. 配置参数 ===== */}
        {columns.length > 0 && (
          <section className="batch-section batch-config">
            <header>
              <div>
                <span>STEP 02</span>
                <h2>配置分类参数</h2>
                <p>选择查询列、检索模式、工作区与分类提示词模板。</p>
              </div>
            </header>
            <div className="batch-config-grid">
              <label className="field">
                <span>查询列</span>
                <select
                  value={queryColumn}
                  onChange={(e) => setQueryColumn(e.target.value)}
                >
                  {columns.map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
                <small>每行该列的内容将作为查询送入 RAG</small>
              </label>
              <label className="field">
                <span>RAG 检索模式</span>
                <select
                  value={mode}
                  onChange={(e) => setMode(e.target.value)}
                >
                  {modes.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name} — {m.summary}
                    </option>
                  ))}
                </select>
                <small>语义检索通常对短文本最稳定</small>
              </label>
              <label className="field">
                <span>检索工作区</span>
                <select
                  value={workspaceId}
                  onChange={(e) => setWorkspaceId(e.target.value)}
                >
                  {workspaces.map((w) => (
                    <option key={w.id} value={w.id}>
                      {w.name} ({w.kind})
                    </option>
                  ))}
                </select>
                <small>默认内置库；可切换到已上传文档工作区</small>
              </label>
              <label className="field">
                <span>Top-K</span>
                <input
                  type="number"
                  min={1}
                  max={20}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                />
                <small>检索返回的证据条数</small>
              </label>
              <label className="field">
                <span>最大处理行数</span>
                <input
                  type="number"
                  min={1}
                  max={5000}
                  value={maxRows}
                  onChange={(e) => setMaxRows(Number(e.target.value))}
                />
                <small>超过的行将被截断，保护 API 配额</small>
              </label>
            </div>
            <label className="batch-prompt-field">
              <span>分类 Prompt 模板</span>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={6}
              />
              <small>
                {"{query} 会被替换为每行评论；{evidence} 会被替换为检索证据片段"}
              </small>
            </label>
          </section>
        )}

        {/* ===== 3. 数据预览 ===== */}
        {previewRows.length > 0 && (
          <section className="batch-section batch-preview">
            <header>
              <div>
                <span>STEP 03</span>
                <h2>数据预览</h2>
                <p>前 10 行用于确认列解析是否正确。带标记的列是查询列。</p>
              </div>
            </header>
            <div className="batch-table-wrap">
              <table className="batch-table preview">
                <thead>
                  <tr>
                    <th>#</th>
                    {columns.map((c) => (
                      <th
                        key={c}
                        className={c === queryColumn ? "is-query-col" : ""}
                      >
                        {c}
                        {c === queryColumn ? " ← 查询" : ""}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {previewRows.map((row, i) => (
                    <tr key={i}>
                      <td className="row-index">{i + 1}</td>
                      {columns.map((c) => (
                        <td
                          key={c}
                          className={c === queryColumn ? "is-query-col" : ""}
                          title={row[c]}
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

        {/* ===== 4. 执行 + 结果 ===== */}
        {columns.length > 0 && (
          <section className="batch-section batch-execute">
            <header>
              <div>
                <span>STEP 04</span>
                <h2>执行批量分类</h2>
                <p>
                  每行将作为独立查询调用 RAG 流水线。耗时取决于行数与 LLM 响应速度。
                </p>
              </div>
            </header>

            <div className="batch-actions">
              <button
                className="build-button"
                onClick={handleRun}
                disabled={loading || !csvContent || !queryColumn}
                type="button"
              >
                {loading ? "正在处理..." : "开始批量分类"}
                <span>→</span>
              </button>
              {result && (
                <button
                  className="batch-download-button"
                  onClick={handleDownload}
                  type="button"
                >
                  下载结果 CSV
                  <span>↓</span>
                </button>
              )}
            </div>

            {!modelReady && !loading && (
              <div className="batch-warning">
                <strong>模型未连接</strong>
                <span>
                  前往「模型连接」页面配置 LLM（DeepSeek 或本地模型），否则分类将回退为模板输出。
                </span>
              </div>
            )}

            {error && (
              <div className="batch-error">
                <strong>执行失败</strong>
                <span>{error}</span>
              </div>
            )}

            {(loading || result) && (
              <section className="ingestion-monitor completed">
                <header>
                  <div>
                    <span>BATCH PIPELINE</span>
                    <strong>
                      {loading
                        ? "正在逐行分类"
                        : result
                          ? "批量分类已完成"
                          : ""}
                    </strong>
                  </div>
                  <b>{progress}%</b>
                </header>
                <div className="progress-rail">
                  <i style={{ width: `${progress}%` }} />
                </div>
                {result && (
                  <div className="ingestion-result-strip">
                    <span>
                      <small>总行数</small>
                      <strong>{result.total_rows}</strong>
                    </span>
                    <span>
                      <small>成功</small>
                      <strong>{result.succeeded}</strong>
                    </span>
                    <span>
                      <small>失败</small>
                      <strong>{result.failed}</strong>
                    </span>
                    <span>
                      <small>查询列</small>
                      <strong>{queryColumn}</strong>
                    </span>
                  </div>
                )}
              </section>
            )}

            {result && result.items.length > 0 && (
              <div className="batch-table-wrap">
                <table className="batch-table results">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>查询内容</th>
                      <th>分类结果</th>
                      <th>理由</th>
                      <th>状态</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.items.map((item) => (
                      <ResultRow key={item.row_index} item={item} />
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

function ResultRow({ item }: { item: BatchClassifyItem }) {
  const label = item.classification || "—";
  const tone =
    label.includes("支持")
      ? "support"
      : label.includes("质疑")
        ? "question"
        : label.includes("噪音")
          ? "noise"
          : "unknown";
  return (
    <tr>
      <td className="row-index">{item.row_index + 1}</td>
      <td className="query-cell" title={item.query_text}>
        {item.query_text}
      </td>
      <td>
        <span className={`classification-pill ${tone}`}>{label}</span>
      </td>
      <td className="reasoning-cell" title={item.reasoning || item.error}>
        {item.reasoning || item.error || "—"}
      </td>
      <td>
        <span className={`status-mark ${item.status}`}>
          {item.status === "success" ? "✓" : item.status === "error" ? "✗" : "…"}
        </span>
      </td>
    </tr>
  );
}
```

---

## 新增的 CSS（追加到 `frontend/app/globals.css` 末尾）

这些 class 与现有 design token 保持一致的色彩、圆角、字号规范：

```css
/* ================================================================
   批量分类页面（/batch）
   继承 .platform-frame / .platform-page / .platform-page-head
   ================================================================ */

.batch-page .batch-section {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 18px;
  margin: 0 8px 22px;
  padding: 24px 28px;
  position: relative;
}

.batch-page .batch-section > header {
  margin-bottom: 18px;
}

.batch-page .batch-section > header > div > span {
  color: var(--accent);
  font-size: 9px;
  font-weight: 800;
  letter-spacing: .2em;
}

.batch-page .batch-section > header > div > h2 {
  font-family: Georgia, "Songti SC", serif;
  font-size: 22px;
  font-weight: 500;
  letter-spacing: -.01em;
  margin: 6px 0 5px;
}

.batch-page .batch-section > header > div > p {
  color: var(--muted);
  font-size: 12px;
  margin: 0;
}

/* 上传区：直接复用 .drop-field，但允许有文件时高亮 */
.batch-page .drop-field.has-file {
  border-color: var(--accent);
  background: #f4f8f5;
}

/* 配置表单 */
.batch-config-grid {
  display: grid;
  gap: 14px 18px;
  grid-template-columns: 1fr 1fr;
}

.batch-config-grid .field > small,
.batch-prompt-field > small {
  color: var(--faint);
  display: block;
  font-size: 10px;
  margin-top: 4px;
}

.batch-prompt-field {
  display: flex;
  flex-direction: column;
  gap: 5px;
  margin-top: 18px;
}

.batch-prompt-field > span {
  color: var(--ink);
  font-size: 11px;
  font-weight: 700;
}

.batch-prompt-field textarea {
  background: #f7f9f8;
  border: 1px solid var(--line-strong);
  border-radius: 9px;
  font-family: "JetBrains Mono", "SF Mono", Consolas, monospace;
  font-size: 12px;
  line-height: 1.55;
  padding: 12px 14px;
  resize: vertical;
}

/* 表格 */
.batch-table-wrap {
  border: 1px solid var(--line);
  border-radius: 12px;
  margin-top: 6px;
  overflow-x: auto;
}

.batch-table {
  border-collapse: collapse;
  font-size: 12px;
  min-width: 100%;
}

.batch-table th {
  background: #f7f9f8;
  border-bottom: 1px solid var(--line);
  color: var(--ink);
  font-size: 10px;
  font-weight: 800;
  letter-spacing: .12em;
  padding: 11px 14px;
  text-align: left;
}

.batch-table td {
  border-bottom: 1px solid var(--line);
  color: var(--ink);
  max-width: 320px;
  overflow: hidden;
  padding: 11px 14px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.batch-table tr:last-child td {
  border-bottom: none;
}

.batch-table .row-index {
  color: var(--faint);
  font-variant-numeric: tabular-nums;
  width: 50px;
}

.batch-table .is-query-col {
  background: #f4f8f5;
  color: var(--accent);
  font-weight: 600;
}

.batch-table.results .query-cell {
  max-width: 280px;
}

.batch-table.results .reasoning-cell {
  color: var(--muted);
  max-width: 380px;
}

/* 分类标签 */
.classification-pill {
  border-radius: 6px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .02em;
  padding: 3px 10px;
  white-space: nowrap;
}

.classification-pill.support {
  background: #e6f3ec;
  color: #2e7d32;
}

.classification-pill.question {
  background: #fdf2e3;
  color: #b86a1c;
}

.classification-pill.noise {
  background: #fbe9ea;
  color: #b1393d;
}

.classification-pill.unknown {
  background: #eef1ee;
  color: var(--muted);
}

/* 状态点 */
.status-mark {
  border-radius: 50%;
  display: inline-block;
  font-size: 12px;
  font-weight: 700;
  height: 22px;
  line-height: 22px;
  text-align: center;
  width: 22px;
}

.status-mark.success {
  background: #e6f3ec;
  color: #2e7d32;
}

.status-mark.error {
  background: #fbe9ea;
  color: #b1393d;
}

.status-mark.pending {
  background: #eef1ee;
  color: var(--muted);
}

/* 按钮 */
.batch-actions {
  display: flex;
  gap: 12px;
  margin-bottom: 18px;
}

.batch-download-button {
  align-items: center;
  background: #fff;
  border: 1px solid var(--accent);
  border-radius: 9px;
  color: var(--accent);
  cursor: pointer;
  display: inline-flex;
  font-size: 13px;
  font-weight: 700;
  gap: 8px;
  padding: 11px 22px;
}

.batch-download-button:hover {
  background: #f4f8f5;
}

.batch-download-button span {
  font-size: 14px;
}

/* 警告 / 错误条 */
.batch-warning,
.batch-error {
  border-radius: 10px;
  display: grid;
  gap: 2px;
  margin-bottom: 18px;
  padding: 12px 16px;
}

.batch-warning {
  background: #fdf2e3;
  border: 1px solid #f3d9aa;
  color: #8a541a;
}

.batch-warning strong {
  color: #6f4316;
  font-size: 12px;
  letter-spacing: .08em;
}

.batch-warning span {
  font-size: 12px;
}

.batch-error {
  background: #fbe9ea;
  border: 1px solid #f3c4c8;
  color: #a53239;
}

.batch-error strong {
  color: #8c2930;
  font-size: 12px;
  letter-spacing: .08em;
}

.batch-error span {
  font-size: 12px;
}

/* 响应式：窄屏单列 */
@media (max-width: 720px) {
  .batch-config-grid {
    grid-template-columns: 1fr;
  }
}
```

---

## 改动清单

| 文件 | 改动 |
|------|------|
| `frontend/app/batch/page.tsx` | **完全重写**：移除所有 inline style，复用 `.platform-frame` / `.platform-page` / `.platform-page-head` / `.workspace-count-card` / `.drop-field` / `.field` / `.ingestion-monitor` / `.progress-rail` / `.ingestion-result-strip` / `.build-button` 等现有 class |
| `frontend/app/globals.css` | **追加** ~250 行 CSS，定义所有 `.batch-*` class 和分类标签/状态点样式 |

## 关键约束

1. **不要保留任何 inline `style={{}}`**（除非是 `style={{ width: `${progress}%` }}` 这种动态值）。所有视觉样式走 CSS class。
2. **必须包 `<div className="platform-frame">`**——这是 PlatformHeader 正常显示的关键。
3. **必须用 `<main className="platform-page batch-page">`**——`platform-page` class 自带 `padding-top: calc(var(--topbar) + 58px)`，自动避开顶栏。
4. **复用 `.drop-field`**——不要自己写上传区，跟数据工作区保持完全一致的拖拽样式。
5. **复用 `.workspace-count-card`**——右上角的暗色卡片显示总行数，跟数据工作区的"WORKSPACES 02"卡片视觉一致。
6. **复用 `.ingestion-monitor` + `.progress-rail` + `.ingestion-result-strip`**——批量执行的进度展示跟数据工作区的"构建流水线"卡片视觉一致。
7. **复用 `.build-button`**——主执行按钮的样式（深色 + 箭头）跟"构建并发布工作区"按钮一致。
8. 分类标签的配色用绿色（支持）/ 橙色（质疑）/ 红色（噪音），保持教学平台的克制风格，不抢眼。