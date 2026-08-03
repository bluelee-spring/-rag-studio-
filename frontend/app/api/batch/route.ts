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
