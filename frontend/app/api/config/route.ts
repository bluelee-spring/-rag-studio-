import { NextResponse } from "next/server";

const backend =
  process.env.RAG_BACKEND_URL ?? "http://127.0.0.1:8000";

export async function GET() {
  try {
    const [modesResponse, summaryResponse, workspacesResponse, modelResponse] = await Promise.all([
      fetch(`${backend}/api/v1/modes`, { cache: "no-store" }),
      fetch(`${backend}/api/v1/dataset/summary`, {
        cache: "no-store",
      }),
      fetch(`${backend}/api/v1/workspaces`, { cache: "no-store" }),
      fetch(`${backend}/api/v1/settings/model`, { cache: "no-store" }),
    ]);

    if (
      !modesResponse.ok ||
      !summaryResponse.ok ||
      !workspacesResponse.ok ||
      !modelResponse.ok
    ) {
      return NextResponse.json(
        { detail: "后端配置接口不可用" },
        { status: 502 },
      );
    }

    return NextResponse.json({
      modes: await modesResponse.json(),
      summary: await summaryResponse.json(),
      workspaces: await workspacesResponse.json(),
      model: await modelResponse.json(),
    });
  } catch {
    return NextResponse.json(
      {
        detail:
          "无法连接FastAPI后端，请确认8000端口已经启动。",
      },
      { status: 502 },
    );
  }
}
