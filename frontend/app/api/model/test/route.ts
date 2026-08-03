import { backendEndpoint, proxyFailure, relayJson } from "@/lib/backend-proxy";

export async function POST() {
  try {
    const response = await fetch(backendEndpoint("/api/v1/settings/model/test"), {
      method: "POST",
      cache: "no-store",
      signal: AbortSignal.timeout(300_000),
    });
    return relayJson(response);
  } catch (error) {
    return proxyFailure(error, "无法连接本地模型测试服务");
  }
}
