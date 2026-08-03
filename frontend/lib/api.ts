import type {
  BatchClassifyRequest,
  BatchClassifyResponse,
  DatasetSummary,
  ModeInfo,
  QueryResponse,
  RagMode,
  IngestionJob,
  ModelConfigInput,
  ModelStatus,
  ModelTestResult,
  WorkspaceInfo,
} from "./types";

export async function loadConfig(): Promise<{
  modes: ModeInfo[];
  summary: DatasetSummary;
  workspaces: WorkspaceInfo[];
  model: ModelStatus;
}> {
  const response = await fetch("/api/config", {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error("无法读取教学系统配置");
  }
  return response.json();
}

export async function runQuery(input: {
  mode: RagMode;
  question: string;
  top_k: number;
  workspace_id: string;
}): Promise<QueryResponse> {
  const response = await fetch("/api/rag", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(input),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "检索执行失败");
  }
  return payload;
}

async function jsonRequest<T>(
  url: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(url, { cache: "no-store", ...init });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "请求执行失败");
  }
  return payload as T;
}

export function listWorkspaces(): Promise<WorkspaceInfo[]> {
  return jsonRequest<WorkspaceInfo[]>("/api/workspaces");
}

export function loadModelStatus(): Promise<ModelStatus> {
  return jsonRequest<ModelStatus>("/api/model");
}

export function saveModelConfig(
  input: ModelConfigInput,
): Promise<ModelStatus> {
  return jsonRequest<ModelStatus>("/api/model", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
}

export function testModelConnection(): Promise<ModelTestResult> {
  return jsonRequest<ModelTestResult>("/api/model/test", {
    method: "POST",
  });
}

export function ingestDocuments(form: FormData): Promise<IngestionJob> {
  return jsonRequest<IngestionJob>("/api/workspaces/documents", {
    method: "POST",
    body: form,
  });
}

export function ingestTable(form: FormData): Promise<IngestionJob> {
  return jsonRequest<IngestionJob>("/api/workspaces/table", {
    method: "POST",
    body: form,
  });
}

export function ingestGraph(form: FormData): Promise<IngestionJob> {
  return jsonRequest<IngestionJob>("/api/workspaces/graph", {
    method: "POST",
    body: form,
  });
}

export function loadIngestionJob(jobId: string): Promise<IngestionJob> {
  return jsonRequest<IngestionJob>(
    `/api/ingestion/jobs/${encodeURIComponent(jobId)}`,
  );
}

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
