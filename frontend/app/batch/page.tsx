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
