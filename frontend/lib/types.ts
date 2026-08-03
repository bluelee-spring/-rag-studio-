export type RagMode =
  | "tfidf"
  | "bm25"
  | "semantic"
  | "property_graph"
  | "rdf"
  | "sql"
  | "composite";

export type ModeInfo = {
  id: RagMode;
  name: string;
  family: string;
  summary: string;
  accent: string;
};

export type DatasetSummary = {
  documents: number;
  chunks: number;
  graph_nodes: number;
  graph_edges: number;
  rdf_triples: number;
  relational_cases: number;
  external: Record<string, boolean>;
};

export type WorkspaceKind = "builtin" | "documents" | "table" | "graph";

export type WorkspaceInfo = {
  id: string;
  name: string;
  kind: WorkspaceKind;
  status: "building" | "ready" | "failed";
  created_at: string;
  updated_at: string;
  source_files: string[];
  supported_modes: RagMode[];
  statistics: Record<string, unknown>;
  build: Record<string, unknown>;
};

export type ModelProvider =
  | "environment"
  | "remote_api"
  | "local_huggingface";

export type ModelStatus = {
  provider: ModelProvider;
  configured: boolean;
  generation_ready: boolean;
  embedding_ready: boolean;
  api_base: string;
  api_key_present: boolean;
  api_key_masked: string;
  chat_model: string;
  embedding_model: string;
  local_model_path: string;
  local_adapter_path: string;
  local_artifact_type:
    | "unknown"
    | "full_model"
    | "lora_adapter"
    | "full_model_with_adapter";
  resolved_model_path: string;
  resolved_adapter_path: string;
  model_loaded: boolean;
  local_device: "auto" | "cpu" | "cuda" | "mps" | string;
  local_dtype: "auto" | "float32" | "float16" | "bfloat16" | string;
  max_new_tokens: number;
  enable_planner: boolean;
  enable_answer: boolean;
  source: string;
  notes: string[];
};

export type ModelConfigInput = {
  provider: "remote_api" | "local_huggingface";
  api_base: string;
  api_key?: string | null;
  chat_model: string;
  embedding_model: string;
  local_model_path: string;
  local_adapter_path: string;
  local_device: "auto" | "cpu" | "cuda" | "mps";
  local_dtype: "auto" | "float32" | "float16" | "bfloat16";
  max_new_tokens: number;
  enable_planner: boolean;
  enable_answer: boolean;
};

export type ModelTestResult = {
  ok: boolean;
  provider: string;
  latency_ms: number;
  message: string;
  details: Record<string, unknown>;
};

export type IngestionStage = {
  id: string;
  label: string;
  status: "pending" | "running" | "completed" | "failed";
  details: Record<string, unknown>;
};

export type IngestionJob = {
  id: string;
  kind: "documents" | "table" | "graph";
  status: "queued" | "running" | "completed" | "failed";
  workspace_id?: string | null;
  progress: number;
  current_stage: string;
  stages: IngestionStage[];
  error: string;
  created_at: string;
  updated_at: string;
};

export type TraceStage = {
  id: string;
  title: string;
  kind: string;
  status: "completed" | "fallback" | "warning";
  description: string;
  duration_ms: number;
  data: Record<string, unknown>;
};

export type EvidenceItem = {
  id: string;
  title: string;
  excerpt: string;
  source: string;
  score?: number | null;
};

export type QueryResponse = {
  run_id: string;
  mode: RagMode;
  mode_name: string;
  question: string;
  answer: string;
  stages: TraceStage[];
  evidence: EvidenceItem[];
  metrics: Record<string, unknown>;
  execution_source: string;
  disclaimer: string;
};

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
