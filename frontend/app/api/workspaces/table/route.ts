import { NextRequest } from "next/server";

import { backendEndpoint, proxyFailure, relayJson } from "@/lib/backend-proxy";

export async function POST(request: NextRequest) {
  try {
    const form = await request.formData();
    const response = await fetch(backendEndpoint("/api/v1/workspaces/table"), {
      method: "POST",
      body: form,
      cache: "no-store",
      signal: AbortSignal.timeout(300_000),
    });
    return relayJson(response);
  } catch (error) {
    return proxyFailure(error, "无法把表文件提交给 FastAPI 建库服务");
  }
}
