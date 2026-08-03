import { NextRequest, NextResponse } from "next/server";

const backend = process.env.RAG_BACKEND_URL ?? "http://127.0.0.1:8000";

export async function GET() {
  try {
    const response = await fetch(`${backend}/api/v1/settings/model`, {
      cache: "no-store",
    });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch {
    return NextResponse.json({ detail: "无法连接模型配置服务" }, { status: 502 });
  }
}

export async function PUT(request: NextRequest) {
  try {
    const response = await fetch(`${backend}/api/v1/settings/model`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(await request.json()),
      cache: "no-store",
    });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch {
    return NextResponse.json({ detail: "无法保存模型配置" }, { status: 502 });
  }
}
