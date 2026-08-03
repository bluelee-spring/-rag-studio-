import { NextResponse } from "next/server";

const backend = process.env.RAG_BACKEND_URL ?? "http://127.0.0.1:8000";

export async function GET() {
  try {
    const response = await fetch(`${backend}/api/v1/workspaces`, {
      cache: "no-store",
    });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch {
    return NextResponse.json({ detail: "无法读取数据工作区" }, { status: 502 });
  }
}
