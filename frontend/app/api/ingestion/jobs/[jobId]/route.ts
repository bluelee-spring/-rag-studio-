import { backendEndpoint, proxyFailure, relayJson } from "@/lib/backend-proxy";

export async function GET(
  _request: Request,
  context: { params: Promise<{ jobId: string }> },
) {
  try {
    const { jobId } = await context.params;
    const response = await fetch(
      backendEndpoint(`/api/v1/ingestion/jobs/${encodeURIComponent(jobId)}`),
      { cache: "no-store" },
    );
    return relayJson(response);
  } catch (error) {
    return proxyFailure(error, "无法读取建库任务状态");
  }
}
