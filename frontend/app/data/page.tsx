"use client";

import Link from "next/link";
import {
  ChangeEvent,
  DragEvent,
  FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { PlatformHeader } from "@/components/PlatformHeader";
import {
  ingestDocuments,
  ingestGraph,
  ingestTable,
  listWorkspaces,
  loadIngestionJob,
  loadModelStatus,
} from "@/lib/api";
import type {
  IngestionJob,
  ModelStatus,
  WorkspaceInfo,
} from "@/lib/types";

type BuilderKind = "documents" | "table" | "graph";

const DOCUMENT_ACCEPT = ".txt,.md,.pdf,.docx,.html,.htm";
const TABLE_ACCEPT = ".csv,.xlsx";
const GRAPH_ACCEPT = ".json,.jsonl";

const GRAPH_CONTAINER_EXAMPLE = `{
  "graph_version": "1.0",
  "name": "领域知识图",
  "schema": {
    "node_types": [{ "name": "Disease", "label": "病害" }],
    "edge_types": [{
      "name": "HAS_SYMPTOM",
      "source_types": ["Disease"],
      "target_types": ["Symptom"]
    }]
  },
  "nodes": [ /* NODE 对象 */ ],
  "edges": [ /* EDGE 对象 */ ],
  "documents": [ /* 可选证据对象 */ ]
}`;

const GRAPH_NODE_EXAMPLE = `{
  "id": "disease_001",
  "type": "Disease",
  "name": "大豆褐斑病",
  "text": "主要危害叶片，可形成褐色近圆形病斑。",
  "properties": { "crop": "大豆" },
  "evidence_ids": ["doc_001"]
}`;

const GRAPH_EDGE_EXAMPLE = `{
  "id": "edge_001",
  "source": "disease_001",
  "target": "symptom_001",
  "type": "HAS_SYMPTOM",
  "text": "该病可表现为褐色近圆形病斑。",
  "properties": { "confidence": 0.96 },
  "evidence_ids": ["doc_001"]
}`;

const GRAPH_DOCUMENT_EXAMPLE = `{
  "id": "doc_001",
  "title": "大豆褐斑病症状说明",
  "text": "大豆褐斑病主要危害叶片，病斑常呈褐色近圆形。",
  "source": "教学资料或原始文件名"
}`;

function readableBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 ** 2).toFixed(1)} MB`;
}

function statistic(workspace: WorkspaceInfo) {
  if (workspace.kind === "documents") {
    return `${Number(workspace.statistics.documents || 0)} 文档 · ${Number(
      workspace.statistics.chunks || 0,
    )} 文本块 · ${Number(workspace.statistics.vectors || 0)} 向量`;
  }
  if (workspace.kind === "table") {
    return `${Number(workspace.statistics.rows || 0)} 行 · ${Number(
      workspace.statistics.columns || 0,
    )} 列 · ${Number(workspace.statistics.indexes || 0)} 索引`;
  }
  if (workspace.kind === "graph") {
    return `${Number(workspace.statistics.nodes || 0)} 节点 · ${Number(
      workspace.statistics.edges || 0,
    )} 边 · ${Number(workspace.statistics.vectors || 0)} 锚点向量`;
  }
  return `${Number(workspace.statistics.chunks || 0)} 文本块 · 图 / RDF / SQL`;
}

function GraphFormatGuide() {
  const [example, setExample] = useState<
    "container" | "node" | "edge" | "document"
  >("container");
  const code =
    example === "container"
      ? GRAPH_CONTAINER_EXAMPLE
      : example === "node"
        ? GRAPH_NODE_EXAMPLE
        : example === "edge"
          ? GRAPH_EDGE_EXAMPLE
          : GRAPH_DOCUMENT_EXAMPLE;
  return (
    <section className="graph-format-guide">
      <header>
        <div>
          <span>GRAPH DATA CONTRACT · 1.0</span>
          <h2>上传前，先确定节点与边</h2>
          <p>系统接收属性图交换格式；不是任意嵌套 JSON，也不是 RDF JSON-LD。</p>
        </div>
        <div className="format-downloads">
          <a href="/examples/graph.json" download>下载 graph.json</a>
          <a href="/examples/graph.jsonl" download>下载 graph.jsonl</a>
        </div>
      </header>
      <div className="graph-contract-grid">
        <div className="contract-structure">
          <div><b>01</b><strong>schema</strong><span>可选；声明节点类型、关系类型和允许方向</span></div>
          <div><b>02</b><strong>nodes</strong><span>必填；每个节点必须具有 id、type、name</span></div>
          <div><b>03</b><strong>edges</strong><span>必填；source、target 必须引用已有节点</span></div>
          <div><b>04</b><strong>documents</strong><span>可选；通过 evidence_ids 为最终回答提供依据</span></div>
        </div>
        <div className="contract-code">
          <nav>
            <button className={example === "container" ? "active" : ""} type="button" onClick={() => setExample("container")}>CONTAINER</button>
            <button className={example === "node" ? "active" : ""} type="button" onClick={() => setExample("node")}>NODE</button>
            <button className={example === "edge" ? "active" : ""} type="button" onClick={() => setExample("edge")}>EDGE</button>
            <button className={example === "document" ? "active" : ""} type="button" onClick={() => setExample("document")}>DOCUMENT</button>
          </nav>
          <pre>{code}</pre>
        </div>
      </div>
      <footer>
        <span><i /> ID在同一图内唯一</span>
        <span><i /> properties必须是对象</span>
        <span><i /> JSONL每行一个record_type</span>
        <span><i /> 证据缺失只警告，悬空边直接拒绝</span>
      </footer>
    </section>
  );
}

function StagePipeline({ job }: { job: IngestionJob }) {
  const completedDetails =
    job.stages.find((stage) => stage.id === "completed")?.details ?? {};
  const validationDetails =
    job.stages.find((stage) => stage.id === "validate-graph")?.details ?? {};
  const warnings = Array.isArray(validationDetails.warnings)
    ? validationDetails.warnings.map(String)
    : [];
  const detailLabels: Record<string, string> = {
    documents: "文档",
    chunks: "文本块",
    vectors: "向量",
    rows: "数据行",
    columns: "字段",
    indexes: "索引",
    nodes: "节点",
    edges: "边",
    node_types: "节点类型",
    edge_types: "关系类型",
    dimensions: "向量维度",
  };
  const summaryEntries = Object.entries(completedDetails).filter(
    ([key, value]) =>
      key in detailLabels && ["string", "number"].includes(typeof value),
  );
  return (
    <section className={`ingestion-monitor ${job.status}`}>
      <header>
        <div>
          <span>BUILD PIPELINE</span>
          <strong>
            {job.status === "completed"
              ? "工作区已发布"
              : job.status === "failed"
                ? "构建中断"
                : "正在构建可查询数据"}
          </strong>
        </div>
        <b>{job.progress}%</b>
      </header>
      <div className="progress-rail">
        <i style={{ width: `${job.progress}%` }} />
      </div>
      <div className="ingestion-stages">
        {job.stages.map((stage, index) => (
          <div className={`ingestion-stage ${stage.status}`} key={stage.id}>
            <span>{String(index + 1).padStart(2, "0")}</span>
            <i />
            <strong>{stage.label}</strong>
            <small>
              {stage.status === "running"
                ? "RUNNING"
                : stage.status === "completed"
                  ? "DONE"
                  : stage.status === "failed"
                    ? "FAILED"
                    : "WAIT"}
            </small>
          </div>
        ))}
      </div>
      {job.status === "completed" && summaryEntries.length > 0 && (
        <div className="ingestion-result-strip">
          {summaryEntries.map(([key, value]) => (
            <span key={key}>
              <small>{detailLabels[key]}</small>
              <strong>{String(value)}</strong>
            </span>
          ))}
        </div>
      )}
      {warnings.length > 0 && (
        <div className="ingestion-warnings">
          <strong>VALIDATION WARNINGS</strong>
          {warnings.map((warning) => <span key={warning}>{warning}</span>)}
        </div>
      )}
      {job.error && <p className="ingestion-error">{job.error}</p>}
      {job.status === "completed" && job.workspace_id && (
        <Link
          className="open-workspace-button"
          href={`/?workspace=${encodeURIComponent(job.workspace_id)}`}
        >
          进入实验台查询
          <span>↗</span>
        </Link>
      )}
    </section>
  );
}

function DropField({
  kind,
  files,
  onFiles,
}: {
  kind: BuilderKind;
  files: File[];
  onFiles: (files: File[]) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const multiple = kind === "documents";
  const acceptTypes =
    kind === "documents" ? DOCUMENT_ACCEPT : kind === "table" ? TABLE_ACCEPT : GRAPH_ACCEPT;

  function accept(next: FileList | null) {
    if (!next) return;
    const selected = Array.from(next);
    onFiles(multiple ? selected.slice(0, 10) : selected.slice(0, 1));
  }

  function drop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    accept(event.dataTransfer.files);
  }

  return (
    <div
      className={dragging ? "drop-field dragging" : "drop-field"}
      onDragEnter={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node)) {
          setDragging(false);
        }
      }}
      onDrop={drop}
      onClick={() => input.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") input.current?.click();
      }}
    >
      <input
        ref={input}
        type="file"
        accept={acceptTypes}
        multiple={multiple}
        onChange={(event: ChangeEvent<HTMLInputElement>) => {
          accept(event.target.files);
          event.currentTarget.value = "";
        }}
      />
      <div className="drop-volume" aria-hidden="true">
        <i />
        <i />
        <i />
        <b>{kind === "documents" ? "DOC" : kind === "table" ? "TAB" : "GPH"}</b>
      </div>
      <strong>{files.length ? "文件已进入暂存区" : "拖拽文件到这里"}</strong>
      <span>
        {kind === "documents"
          ? "TXT · Markdown · PDF · Word · HTML，最多 10 个"
          : kind === "table"
            ? "CSV 或 XLSX，一次生成一个关系数据库"
            : "graph.json 或 graph.jsonl，一次生成一个属性图工作区"}
      </span>
      <small>单文件上限 25 MB · 单次上限 100 MB</small>
    </div>
  );
}

export default function DataPage() {
  const [kind, setKind] = useState<BuilderKind>("documents");
  const [name, setName] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [chunkSize, setChunkSize] = useState(700);
  const [chunkOverlap, setChunkOverlap] = useState(100);
  const [sheetName, setSheetName] = useState("");
  const [workspaces, setWorkspaces] = useState<WorkspaceInfo[]>([]);
  const [model, setModel] = useState<ModelStatus | null>(null);
  const [job, setJob] = useState<IngestionJob | null>(null);
  const [message, setMessage] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const refresh = useCallback(() => {
    Promise.all([listWorkspaces(), loadModelStatus()])
      .then(([nextWorkspaces, nextModel]) => {
        setWorkspaces(nextWorkspaces);
        setModel(nextModel);
      })
      .catch((error) => setMessage(error.message));
  }, []);

  useEffect(() => refresh(), [refresh]);

  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    let active = true;
    const timer = window.setInterval(async () => {
      try {
        const next = await loadIngestionJob(job.id);
        if (!active) return;
        setJob(next);
        if (next.status === "completed") refresh();
      } catch (error) {
        if (active) setMessage(error instanceof Error ? error.message : "无法读取构建状态");
      }
    }, 900);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [job, refresh]);

  const customWorkspaces = useMemo(
    () => workspaces.filter((workspace) => workspace.kind !== "builtin"),
    [workspaces],
  );

  function switchKind(next: BuilderKind) {
    if (submitting) return;
    setKind(next);
    setFiles([]);
    setJob(null);
    setMessage("");
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!name.trim()) {
      setMessage("请先填写工作区名称");
      return;
    }
    if (!files.length) {
      setMessage("请先选择要处理的文件");
      return;
    }
    setSubmitting(true);
    setMessage("");
    setJob(null);
    try {
      const form = new FormData();
      form.set("name", name.trim());
      if (kind === "documents") {
        form.set("chunk_size", String(chunkSize));
        form.set("chunk_overlap", String(chunkOverlap));
        files.forEach((file) => form.append("files", file));
        setJob(await ingestDocuments(form));
      } else if (kind === "table") {
        form.set("sheet_name", sheetName.trim());
        form.set("file", files[0]);
        setJob(await ingestTable(form));
      } else {
        form.set("file", files[0]);
        setJob(await ingestGraph(form));
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "提交失败");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="platform-frame">
      <PlatformHeader serviceOnline={Boolean(model)} modelReady={model?.generation_ready} />
      <main className="platform-page data-page">
        <header className="platform-page-head">
          <div>
            <span>DATA WORKSPACE</span>
            <h1>数据工作区</h1>
            <p>原始文件进入后端流水线，发布为可直接检索的独立工作区。</p>
          </div>
          <div className="workspace-count-card">
            <span>WORKSPACES</span>
            <strong>{String(customWorkspaces.length).padStart(2, "0")}</strong>
            <small>本机应用数据</small>
          </div>
        </header>

        <section className="data-builder">
          <div className="builder-tabs">
            <button
              className={kind === "documents" ? "active" : ""}
              onClick={() => switchKind("documents")}
              type="button"
            >
              <span>DOCUMENT</span>
              <strong>文档检索库</strong>
              <small>TF–IDF · BM25 · Semantic / FAISS</small>
            </button>
            <button
              className={kind === "table" ? "active" : ""}
              onClick={() => switchKind("table")}
              type="button"
            >
              <span>RELATIONAL</span>
              <strong>关系数据库</strong>
              <small>Schema inference · SQLite · SQL RAG</small>
            </button>
            <button
              className={kind === "graph" ? "active" : ""}
              onClick={() => switchKind("graph")}
              type="button"
            >
              <span>PROPERTY GRAPH</span>
              <strong>图数据工作区</strong>
              <small>JSON / JSONL · SQLite Graph · FAISS</small>
            </button>
          </div>

          {kind === "graph" && <GraphFormatGuide />}

          <form className="builder-console" onSubmit={submit}>
            <div className="builder-input-zone">
              <DropField kind={kind} files={files} onFiles={setFiles} />
              {files.length > 0 && (
                <div className="file-queue">
                  {files.map((file, index) => (
                    <div key={`${file.name}-${file.lastModified}`}>
                      <span>{String(index + 1).padStart(2, "0")}</span>
                      <strong>{file.name}</strong>
                      <small>{readableBytes(file.size)}</small>
                      <button
                        type="button"
                        aria-label={`移除${file.name}`}
                        onClick={() => setFiles((current) => current.filter((_, item) => item !== index))}
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="builder-parameters">
              <label className="field">
                <span>工作区名称</span>
                <input
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder={
                    kind === "documents"
                      ? "例如：水稻病害手册"
                      : kind === "table"
                        ? "例如：田间巡检记录"
                        : "例如：水稻病害知识图"
                  }
                  maxLength={100}
                />
              </label>
              {kind === "documents" ? (
                <>
                  <label className="field">
                    <span>目标分块长度</span>
                    <input
                      type="number"
                      min={300}
                      max={2000}
                      value={chunkSize}
                      onChange={(event) => setChunkSize(Number(event.target.value))}
                    />
                    <small>按段落和句子边界自动切分，超长句才按字符截断</small>
                  </label>
                  <label className="field">
                    <span>相邻重叠长度</span>
                    <input
                      type="number"
                      min={0}
                      max={Math.min(500, chunkSize - 1)}
                      value={chunkOverlap}
                      onChange={(event) => setChunkOverlap(Number(event.target.value))}
                    />
                    <small>保留跨块语义连续性，必须小于目标分块长度</small>
                  </label>
                  <div className="artifact-map">
                    <span>原文</span><i />
                    <span>Chunk</span><i />
                    <span>词法索引</span><i />
                    <span>FAISS</span>
                  </div>
                </>
              ) : kind === "table" ? (
                <>
                  <label className="field">
                    <span>Excel 工作表（可选）</span>
                    <input
                      value={sheetName}
                      onChange={(event) => setSheetName(event.target.value)}
                      placeholder="留空读取第一个工作表"
                    />
                  </label>
                  <div className="artifact-map table-map">
                    <span>表头</span><i />
                    <span>类型推断</span><i />
                    <span>SQLite</span><i />
                    <span>只读 SQL</span>
                  </div>
                </>
              ) : (
                <>
                  <div className="graph-build-boundary">
                    <span>执行边界</span>
                    <strong>最多 50,000 节点 · 200,000 边</strong>
                    <small>查询固定为关系白名单、双向邻接和最多 2 跳局部扩展</small>
                  </div>
                  <div className="artifact-map graph-map">
                    <span>校验</span><i />
                    <span>SQLite 图</span><i />
                    <span>节点语义卡</span><i />
                    <span>FAISS</span>
                  </div>
                </>
              )}
              <button className="build-button" disabled={submitting} type="submit">
                {submitting ? "正在接收文件" : "构建并发布工作区"}
                <span>→</span>
              </button>
              {(kind === "documents" || kind === "graph") && (
                <div className={model?.embedding_ready ? "index-provider-note remote" : "index-provider-note"}>
                  <i />
                  <span>
                    {model?.embedding_ready
                      ? `Embedding：${model.embedding_model}；${kind === "graph" ? "节点语义卡" : "文本块"}会发送到已配置的远程端点。`
                      : `Embedding：192维确定性本地编码器；${kind === "graph" ? "图数据" : "文档内容"}不离开本机。`}
                  </span>
                </div>
              )}
              {message && <p className="builder-message">{message}</p>}
            </div>
          </form>
        </section>

        {job && <StagePipeline job={job} />}

        <section className="workspace-catalog">
          <header>
            <div><span>CATALOG</span><h2>已发布工作区</h2></div>
            <small>每个工作区拥有独立文件、索引与数据库</small>
          </header>
          {customWorkspaces.length ? (
            <div className="workspace-grid">
              {customWorkspaces.map((workspace) => (
                <article className={`workspace-card ${workspace.kind}`} key={workspace.id}>
                  <header>
                    <span>{workspace.kind === "documents" ? "DOCUMENT" : workspace.kind === "table" ? "RELATIONAL" : "PROPERTY GRAPH"}</span>
                    <i />
                    <small>READY</small>
                  </header>
                  <h3>{workspace.name}</h3>
                  <p>{statistic(workspace)}</p>
                  <div className="workspace-modes">
                    {workspace.supported_modes.map((mode) => <span key={mode}>{mode}</span>)}
                  </div>
                  <footer>
                    <small title={workspace.source_files.join("、")}>{workspace.source_files.join(" · ")}</small>
                    <Link href={`/?workspace=${encodeURIComponent(workspace.id)}`}>打开 <span>↗</span></Link>
                  </footer>
                </article>
              ))}
            </div>
          ) : (
            <div className="empty-catalog">
              <i />
              <strong>还没有应用数据工作区</strong>
              <span>完成上方任一构建流水线后，它会在这里出现。</span>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
