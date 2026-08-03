import { NextRequest, NextResponse } from "next/server";

const backend =
  process.env.RAG_BACKEND_URL ?? "http://127.0.0.1:8000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const upstream = await fetch(`${backend}/api/v1/rag/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      cache: "no-store",
      // 首次本地推理还包含权重装载，课堂电脑需要更长的首包预算。
      signal: AbortSignal.timeout(300_000),
    });

    const payload = await upstream.json();
    return NextResponse.json(payload, {
      status: upstream.status,
    });
  } catch (error) {
    const detail =
      error instanceof Error && error.name === "TimeoutError"
        ? "后端执行超过300秒"
        : "无法连接FastAPI后端";
    return NextResponse.json({ detail }, { status: 502 });
  }
}
