from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RagMode = Literal[
    "tfidf",
    "bm25",
    "semantic",
    "property_graph",
    "rdf",
    "sql",
    "composite",
]


class QueryRequest(BaseModel):
    mode: RagMode
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    workspace_id: str = Field(
        default="builtin-soybean",
        min_length=3,
        max_length=80,
        pattern=r"^[a-z0-9][a-z0-9-]*$",
    )


class TraceStage(BaseModel):
    id: str
    title: str
    kind: str
    status: Literal["completed", "fallback", "warning"] = "completed"
    description: str = ""
    duration_ms: int = 0
    data: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    id: str
    title: str
    excerpt: str = ""
    source: str = "教学模拟知识库"
    score: float | None = None


class QueryResponse(BaseModel):
    run_id: str
    mode: RagMode
    mode_name: str
    question: str
    answer: str
    stages: list[TraceStage]
    evidence: list[EvidenceItem] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    execution_source: str = "embedded"
    disclaimer: str = (
        "全部病例、药剂与安全间隔期均为教学模拟数据，"
        "不能用于真实诊断、用药或安全判断。"
    )


class ModeInfo(BaseModel):
    id: RagMode
    name: str
    family: str
    summary: str
    accent: str


class DatasetSummary(BaseModel):
    documents: int
    chunks: int
    graph_nodes: int
    graph_edges: int
    rdf_triples: int
    relational_cases: int
    external: dict[str, bool]


WorkspaceKind = Literal["builtin", "documents", "table", "graph"]


class WorkspaceInfo(BaseModel):
    id: str
    name: str
    kind: WorkspaceKind
    status: Literal["building", "ready", "failed"] = "ready"
    created_at: str
    updated_at: str
    source_files: list[str] = Field(default_factory=list)
    supported_modes: list[RagMode] = Field(default_factory=list)
    statistics: dict[str, Any] = Field(default_factory=dict)
    build: dict[str, Any] = Field(default_factory=dict)


ModelProvider = Literal["environment", "remote_api", "local_huggingface"]


class ModelConfigRequest(BaseModel):
    provider: Literal["remote_api", "local_huggingface"]
    api_base: str = Field(default="", max_length=1000)
    api_key: str | None = Field(default=None, max_length=4000)
    chat_model: str = Field(default="", max_length=300)
    embedding_model: str = Field(default="", max_length=300)
    local_model_path: str = Field(default="", max_length=2000)
    local_adapter_path: str = Field(default="", max_length=2000)
    local_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    local_dtype: Literal[
        "auto", "float32", "float16", "bfloat16"
    ] = "auto"
    max_new_tokens: int = Field(default=512, ge=32, le=4096)
    enable_planner: bool = False
    enable_answer: bool = True


class ModelPublicStatus(BaseModel):
    provider: ModelProvider
    configured: bool
    generation_ready: bool
    embedding_ready: bool
    api_base: str = ""
    api_key_present: bool = False
    api_key_masked: str = ""
    chat_model: str = ""
    embedding_model: str = ""
    local_model_path: str = ""
    local_adapter_path: str = ""
    local_artifact_type: Literal[
        "unknown",
        "full_model",
        "lora_adapter",
        "full_model_with_adapter",
    ] = "unknown"
    resolved_model_path: str = ""
    resolved_adapter_path: str = ""
    model_loaded: bool = False
    local_device: str = "auto"
    local_dtype: str = "auto"
    max_new_tokens: int = 512
    enable_planner: bool = False
    enable_answer: bool = True
    source: str = "runtime"
    notes: list[str] = Field(default_factory=list)


class ModelTestResponse(BaseModel):
    ok: bool
    provider: str
    latency_ms: int
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class IngestionJobResponse(BaseModel):
    id: str
    kind: Literal["documents", "table", "graph"]
    status: Literal["queued", "running", "completed", "failed"]
    workspace_id: str | None = None
    progress: int = Field(ge=0, le=100)
    current_stage: str
    stages: list[dict[str, Any]] = Field(default_factory=list)
    error: str = ""
    created_at: str
    updated_at: str


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
