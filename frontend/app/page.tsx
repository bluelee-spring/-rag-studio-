"use client";

import {
  FormEvent,
  KeyboardEvent,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

import { PipelinePlayer } from "@/components/PipelinePlayer";
import { PlatformHeader } from "@/components/PlatformHeader";
import { loadConfig, runQuery } from "@/lib/api";
import type {
  DatasetSummary,
  ModelStatus,
  ModeInfo,
  QueryResponse,
  RagMode,
  WorkspaceInfo,
} from "@/lib/types";

const FALLBACK_MODES: ModeInfo[] = [
  {
    id: "tfidf",
    name: "TF–IDF",
    family: "文档检索",
    summary: "词频、逆文档频率与余弦相似度",
    accent: "#2F6B57",
  },
  {
    id: "bm25",
    name: "BM25",
    family: "文档检索",
    summary: "词频饱和、长度归一化与稀疏排名",
    accent: "#86642F",
  },
  {
    id: "semantic",
    name: "语义向量",
    family: "文档检索",
    summary: "Embedding、FAISS与语义近邻",
    accent: "#66558B",
  },
  {
    id: "property_graph",
    name: "属性图",
    family: "图检索",
    summary: "实体锚点、Cypher与局部子图",
    accent: "#315F76",
  },
  {
    id: "rdf",
    name: "RDF / SPARQL",
    family: "图检索",
    summary: "IRI、本体词汇与证据三元组",
    accent: "#8B4F4B",
  },
  {
    id: "sql",
    name: "关系数据库",
    family: "结构化查询",
    summary: "查询计划、表连接与精确聚合",
    accent: "#4E5B66",
  },
  {
    id: "composite",
    name: "综合 RAG",
    family: "任务编排",
    summary: "多检索器路由、证据合并与LLM生成",
    accent: "#1F493D",
  },
];

const EMPTY_SUMMARY: DatasetSummary = {
  documents: 0,
  chunks: 0,
  graph_nodes: 0,
  graph_edges: 0,
  rdf_triples: 0,
  relational_cases: 0,
  external: {},
};

const QUESTIONS: Record<RagMode, string> = {
  tfidf: "大豆叶片出现褐色近圆形病斑，最可能关联哪些病害？",
  bm25: "大豆叶片出现褐色近圆形病斑，最可能关联哪些病害？",
  semantic: "大豆叶面长出接近圆形的棕褐色斑点，可能是哪种病？",
  property_graph:
    "2025年7月六合区的大豆病例中，叶片出现褐色近圆形病斑时，最可能是什么病？还常伴随哪些症状？",
  rdf: "大豆褐斑病还伴随哪些症状？推荐一种安全间隔期不超过14天的药剂，并给出依据。",
  sql: "2025年7月六合区的大豆病例中，叶片出现褐色近圆形病斑时，最可能是什么病？共有多少例？还常伴随哪些症状？",
  composite:
    "2025年7月六合区的大豆病例中，叶片出现褐色近圆形病斑时，最可能是什么病？共有多少例？还常伴随哪些症状？推荐一种安全间隔期不超过14天的药剂，并给出依据。",
};

const FAMILY_ORDER = ["文档检索", "图检索", "结构化查询", "任务编排"];

function WelcomeFlow({ mode }: { mode: ModeInfo }) {
  return (
    <div className="welcome-flow" aria-label="问题经过检索后进入LLM生成回答">
      <div className="welcome-node question-node">
        <i>?</i>
        <span>用户问题</span>
      </div>
      <div className="welcome-link">
        <i />
        <b />
      </div>
      <div className="welcome-node retrieval-node">
        <i>R</i>
        <span>{mode.name}</span>
      </div>
      <div className="welcome-link">
        <i />
        <b />
      </div>
      <div className="welcome-node context-node">
        <i>≡</i>
        <span>证据上下文</span>
      </div>
      <div className="welcome-link">
        <i />
        <b />
      </div>
      <div className="welcome-node model-node">
        <div className="model-rings">
          <i />
          <i />
        </div>
        <span>LLM</span>
      </div>
      <div className="welcome-answer">
        <i />
        <i />
        <i />
      </div>
    </div>
  );
}

function ThinkingFlow({ mode }: { mode: ModeInfo }) {
  return (
    <div className="thinking-flow" aria-live="polite">
      <div className="thinking-orb">
        <i />
        <i />
        <i />
      </div>
      <div>
        <strong>正在运行 {mode.name}</strong>
        <span>解析问题 · 检索证据 · 组织上下文</span>
      </div>
      <div className="thinking-track">
        <i />
      </div>
    </div>
  );
}

export default function Home() {
  const [modes, setModes] = useState<ModeInfo[]>(FALLBACK_MODES);
  const [summary, setSummary] = useState<DatasetSummary>(EMPTY_SUMMARY);
  const [workspaces, setWorkspaces] = useState<WorkspaceInfo[]>([]);
  const [model, setModel] = useState<ModelStatus | null>(null);
  const [selectedWorkspaceId, setSelectedWorkspaceId] = useState("builtin-soybean");
  const [selectedMode, setSelectedMode] = useState<RagMode>("composite");
  const [question, setQuestion] = useState(QUESTIONS.composite);
  const [submittedQuestion, setSubmittedQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [playbackIndex, setPlaybackIndex] = useState(0);
  const [displayedAnswer, setDisplayedAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [configOnline, setConfigOnline] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    loadConfig()
      .then((config) => {
        if (!active) return;
        setModes(config.modes);
        setSummary(config.summary);
        setWorkspaces(config.workspaces);
        setModel(config.model);
        const requested = new URLSearchParams(window.location.search).get("workspace");
        const workspace = config.workspaces.find((item) => item.id === requested);
        if (workspace) {
          const nextMode: RagMode =
            workspace.kind === "documents"
              ? "semantic"
              : workspace.kind === "table"
                ? "sql"
                : workspace.kind === "graph"
                  ? "property_graph"
                  : "composite";
          setSelectedWorkspaceId(workspace.id);
          setSelectedMode(nextMode);
          if (workspace.kind !== "builtin") setQuestion("");
        }
        setConfigOnline(true);
      })
      .catch(() => {
        if (!active) return;
        setConfigOnline(false);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (
      !response ||
      playbackIndex < response.stages.length - 1
    ) {
      setDisplayedAnswer("");
      return;
    }
    let cursor = 0;
    setDisplayedAnswer("");
    const timer = window.setInterval(() => {
      cursor = Math.min(response.answer.length, cursor + 2);
      setDisplayedAnswer(response.answer.slice(0, cursor));
      if (cursor >= response.answer.length) window.clearInterval(timer);
    }, 28);
    return () => window.clearInterval(timer);
  }, [playbackIndex, response]);

  const selected = useMemo(
    () =>
      modes.find((mode) => mode.id === selectedMode) ??
      FALLBACK_MODES.find((mode) => mode.id === selectedMode)!,
    [modes, selectedMode],
  );

  const selectedWorkspace = useMemo(
    () => workspaces.find((workspace) => workspace.id === selectedWorkspaceId),
    [selectedWorkspaceId, workspaces],
  );

  const availableModes = useMemo(() => {
    if (!selectedWorkspace) return modes;
    const allowed = new Set(selectedWorkspace.supported_modes);
    return modes.filter((mode) => allowed.has(mode.id));
  }, [modes, selectedWorkspace]);

  const groupedModes = useMemo(
    () =>
      FAMILY_ORDER.map((family) => ({
        family,
        items: availableModes.filter((mode) => mode.family === family),
      })).filter((group) => group.items.length > 0),
    [availableModes],
  );

  const handleStageChange = useCallback((index: number) => {
    setPlaybackIndex(index);
  }, []);

  function chooseMode(mode: RagMode) {
    setSelectedMode(mode);
    if (selectedWorkspaceId === "builtin-soybean") {
      setQuestion(QUESTIONS[mode]);
    }
    setSubmittedQuestion("");
    setResponse(null);
    setDisplayedAnswer("");
    setError("");
  }

  function chooseWorkspace(workspaceId: string) {
    const workspace = workspaces.find((item) => item.id === workspaceId);
    if (!workspace) return;
    const nextMode: RagMode =
      workspace.kind === "documents"
        ? "semantic"
        : workspace.kind === "table"
          ? "sql"
          : workspace.kind === "graph"
            ? "property_graph"
            : "composite";
    setSelectedWorkspaceId(workspace.id);
    setSelectedMode(nextMode);
    setQuestion(workspace.kind === "builtin" ? QUESTIONS[nextMode] : "");
    setSubmittedQuestion("");
    setResponse(null);
    setDisplayedAnswer("");
    setError("");
    const url = new URL(window.location.href);
    if (workspace.kind === "builtin") url.searchParams.delete("workspace");
    else url.searchParams.set("workspace", workspace.id);
    window.history.replaceState({}, "", url);
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!question.trim() || loading) return;
    const nextQuestion = question.trim();
    setSubmittedQuestion(nextQuestion);
    setLoading(true);
    setError("");
    setResponse(null);
    setDisplayedAnswer("");
    setPlaybackIndex(0);
    try {
      const result = await runQuery({
        mode: selectedMode,
        question: nextQuestion,
        top_k: topK,
        workspace_id: selectedWorkspaceId,
      });
      setResponse(result);
      setConfigOnline(true);
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : "检索执行失败，请检查后端服务",
      );
    } finally {
      setLoading(false);
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  const generationStage = response?.stages.find(
    (stage) => stage.kind === "generation",
  );
  const llmEnabled = Boolean(generationStage?.data.llm_enabled);
  const answerReady =
    Boolean(response) &&
    playbackIndex >= (response?.stages.length ?? 1) - 1;

  return (
    <div
      className="app-frame"
      style={{ "--accent": selected.accent } as React.CSSProperties}
    >
      <PlatformHeader
        serviceOnline={configOnline}
        modelReady={model?.generation_ready}
      />

      <aside className="mode-sidebar" aria-label="RAG方法">
        <div className="workspace-selector">
          <span>DATA WORKSPACE</span>
          <label>
            <i className={selectedWorkspace?.kind || "builtin"} />
            <select
              aria-label="选择数据工作区"
              value={selectedWorkspaceId}
              onChange={(event) => chooseWorkspace(event.target.value)}
            >
              {workspaces.length === 0 && (
                <option value="builtin-soybean">内置大豆教学库</option>
              )}
              {workspaces.map((workspace) => (
                <option key={workspace.id} value={workspace.id}>
                  {workspace.name}
                </option>
              ))}
            </select>
          </label>
          <small>
            {selectedWorkspace?.kind === "documents"
              ? "独立词法索引与向量索引"
              : selectedWorkspace?.kind === "table"
                ? "独立 SQLite 与字段映射"
                : selectedWorkspace?.kind === "graph"
                  ? "独立 SQLite 属性图与节点 FAISS 索引"
                  : "内置跨范式教学数据"}
          </small>
        </div>
        <div className="sidebar-intro">
          <span>METHOD</span>
          <p>选择检索方式</p>
        </div>
        <nav>
          {groupedModes.map((group) => (
            <section className="mode-group" key={group.family}>
              <h2>{group.family}</h2>
              {group.items.map((mode) => (
                <button
                  className={
                    selectedMode === mode.id
                      ? "mode-button active"
                      : "mode-button"
                  }
                  key={mode.id}
                  onClick={() => chooseMode(mode.id)}
                  type="button"
                >
                  <i style={{ background: mode.accent }} />
                  <span>
                    <strong>{mode.name}</strong>
                    <small>{mode.summary}</small>
                  </span>
                </button>
              ))}
            </section>
          ))}
        </nav>
        <div className="sidebar-foot">
          <span>{selectedWorkspace?.name || "教学模拟数据"}</span>
          <small>
            {selectedWorkspace?.kind === "documents"
              ? `${Number(selectedWorkspace.statistics.chunks || 0)} 文本块 · ${Number(selectedWorkspace.statistics.vectors || 0)} 向量`
              : selectedWorkspace?.kind === "table"
                ? `${Number(selectedWorkspace.statistics.rows || 0)} 行 · ${Number(selectedWorkspace.statistics.columns || 0)} 列`
                : selectedWorkspace?.kind === "graph"
                  ? `${Number(selectedWorkspace.statistics.nodes || 0)} 节点 · ${Number(selectedWorkspace.statistics.edges || 0)} 边`
                  : `${summary.chunks} 文本块 · ${summary.relational_cases} 病例`}
          </small>
        </div>
      </aside>

      <main className="chat-main" id="top">
        <div className="chat-thread">
          {!submittedQuestion && (
            <section className="welcome-state">
              <span className="welcome-kicker">{selected.family}</span>
              <h1>看见一次 RAG 如何完成回答</h1>
              <p>
                <strong>{selectedWorkspace?.name || "内置大豆教学库"}</strong>
                <i> / </i>
                当前方法：<strong>{selected.name}</strong>
              </p>
              <WelcomeFlow mode={selected} />
            </section>
          )}

          {submittedQuestion && (
            <article className="user-turn">
              <div className="turn-avatar user-avatar">你</div>
              <div className="user-bubble">{submittedQuestion}</div>
            </article>
          )}

          {loading && (
            <article className="assistant-turn">
              <div className="turn-avatar assistant-avatar">R</div>
              <div className="assistant-body">
                <ThinkingFlow mode={selected} />
              </div>
            </article>
          )}

          {error && (
            <article className="assistant-turn">
              <div className="turn-avatar assistant-avatar">R</div>
              <div className="assistant-body">
                <div className="error-banner">
                  <strong>执行中断</strong>
                  <span>{error}</span>
                </div>
              </div>
            </article>
          )}

          {response && (
            <article className="assistant-turn">
              <div className="turn-avatar assistant-avatar">R</div>
              <div className="assistant-body">
                <header className="assistant-run-head">
                  <div>
                    <strong>{response.mode_name}</strong>
                    <span>{response.execution_source}</span>
                  </div>
                  <span className={llmEnabled ? "llm-badge active" : "llm-badge"}>
                    {llmEnabled
                      ? `${String(generationStage?.data.provider)} 生成`
                      : "本地模板代替 LLM"}
                  </span>
                </header>

                <PipelinePlayer
                  stages={response.stages}
                  onStageChange={handleStageChange}
                />

                <section
                  className={
                    answerReady
                      ? "generated-answer visible"
                      : "generated-answer"
                  }
                >
                  <header>
                    <span>最终回答</span>
                    <small>
                      {llmEnabled ? "LLM + 检索证据" : "结构化证据模板"}
                    </small>
                  </header>
                  {answerReady ? (
                    <p>
                      {displayedAnswer}
                      {displayedAnswer.length < response.answer.length && (
                        <i className="answer-caret" />
                      )}
                    </p>
                  ) : (
                    <div className="answer-waiting">
                      回答将在检索证据进入生成阶段后出现
                    </div>
                  )}
                </section>

                {answerReady && response.evidence.length > 0 && (
                  <details className="source-drawer">
                    <summary>
                      查看进入上下文的证据
                      <span>{response.evidence.length}</span>
                    </summary>
                    <div className="source-grid">
                      {response.evidence.slice(0, 8).map((item, index) => (
                        <article key={`${item.id}-${index}`}>
                          <span>{String(index + 1).padStart(2, "0")}</span>
                          <div>
                            <strong>{item.title}</strong>
                            <p>{item.excerpt}</p>
                            <small>{item.id}</small>
                          </div>
                        </article>
                      ))}
                    </div>
                  </details>
                )}
                {answerReady && (
                  <p className="disclaimer">{response.disclaimer}</p>
                )}
              </div>
            </article>
          )}
        </div>

        <form className="chat-composer" onSubmit={handleSubmit}>
          <textarea
            id="question"
            aria-label="输入问题"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={handleComposerKeyDown}
            rows={2}
            maxLength={2000}
            placeholder="输入一个问题"
            spellCheck={false}
          />
          <div className="composer-footer">
            <div className="composer-options">
              <span className="selected-method">
                <i style={{ background: selected.accent }} />
                {selected.name}
              </span>
              <label>
                <span>Top K</span>
                <select
                  value={topK}
                  onChange={(event) => setTopK(Number(event.target.value))}
                >
                  {[3, 5, 8, 10].map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </label>
              <button
                type="button"
                className="example-button"
                onClick={() =>
                  setQuestion(
                    selectedWorkspaceId === "builtin-soybean"
                      ? QUESTIONS[selectedMode]
                      : "",
                  )
                }
              >
                {selectedWorkspaceId === "builtin-soybean" ? "示例问题" : "清空问题"}
              </button>
            </div>
            <span className="composer-hint">Enter 发送 · Shift + Enter 换行</span>
            <button
              className="send-button"
              type="submit"
              disabled={loading || !question.trim()}
              aria-label="发送问题"
            >
              {loading ? <i className="spinner" /> : "↑"}
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}
