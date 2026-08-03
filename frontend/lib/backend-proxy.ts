import "server-only";

import { NextResponse } from "next/server";

const backend = (process.env.RAG_BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

export function backendEndpoint(path: string) {
  return `${backend}${path.startsWith("/") ? path : `/${path}`}`;
}

export async function relayJson(response: Response) {
  const text = await response.text();
  let payload: unknown;
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    payload = {
      detail: text || `FastAPI 返回了无法解析的响应（HTTP ${response.status}）`,
    };
  }
  return NextResponse.json(payload, { status: response.status });
}

export function proxyFailure(error: unknown, fallback: string) {
  const timedOut =
    error instanceof Error &&
    (error.name === "TimeoutError" || error.name === "AbortError");
  return NextResponse.json(
    { detail: timedOut ? "FastAPI 执行超时，请查看后端窗口" : fallback },
    { status: 502 },
  );
}
