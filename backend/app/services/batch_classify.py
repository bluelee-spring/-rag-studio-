from __future__ import annotations

import csv
import io
import json

from ..models import (
    BatchClassifyItem,
    BatchClassifyRequest,
    BatchClassifyResponse,
    EvidenceItem,
    QueryRequest,
    TraceStage,
)
from .model_runtime import model_runtime
from .orchestrator import RagOrchestrator
from .workspace_query import WorkspaceRagRouter


class BatchClassifyService:
    """逐行读取 CSV，将指定列作为 query 传入 RAG 检索，再用自定义 prompt 调 LLM 生成分类。"""

    def __init__(
        self,
        orchestrator: RagOrchestrator,
        workspace_router: WorkspaceRagRouter,
    ) -> None:
        self._orchestrator = orchestrator
        self._workspace_router = workspace_router

    def run(self, request: BatchClassifyRequest) -> BatchClassifyResponse:
        # 1. 解析 CSV
        reader = csv.DictReader(io.StringIO(request.csv_content))
        if not reader.fieldnames:
            raise ValueError("CSV 文件没有列头")
        columns = list(reader.fieldnames)
        if request.query_column not in columns:
            raise ValueError(
                f"查询列 '{request.query_column}' 不存在于 CSV 列头中。"
                f"可用列：{columns}"
            )

        rows = list(reader)
        if len(rows) > request.max_rows:
            rows = rows[: request.max_rows]

        items: list[BatchClassifyItem] = []
        succeeded = 0
        failed = 0

        for idx, row in enumerate(rows):
            query_text = (row.get(request.query_column) or "").strip()
            if not query_text:
                items.append(BatchClassifyItem(
                    row_index=idx,
                    raw_data={k: str(v) for k, v in row.items()},
                    query_text="",
                    status="error",
                    error="查询列为空",
                ))
                failed += 1
                continue

            item = BatchClassifyItem(
                row_index=idx,
                raw_data={k: str(v) for k, v in row.items()},
                query_text=query_text,
            )

            try:
                # 2. 调用现有 RAG 检索流水线（复用 orchestrator / workspace_router）
                qr = QueryRequest(
                    mode=request.mode,
                    question=query_text[:2000],
                    top_k=request.top_k,
                    workspace_id=request.workspace_id,
                )
                if request.workspace_id != "builtin-soybean":
                    resp = self._workspace_router.run(qr)
                else:
                    resp = self._orchestrator.run(qr)

                # 3. 提取检索证据摘要
                evidence_text = ""
                if resp.evidence:
                    evidence_text = " | ".join(
                        f"[{e.title}] {e.excerpt[:200]}" for e in resp.evidence[:3]
                    )
                item.evidence_snippet = evidence_text[:500]

                # 4. 用自定义 prompt 直接调 LLM 生成分类
                #    （绕过 AnswerGenerator，因为它的大豆病害 prompt 不适用于分类场景）
                if model_runtime.answer_enabled:
                    user_prompt = request.classification_prompt.replace(
                        "{query}", query_text[:2000]
                    ).replace(
                        "{evidence}", evidence_text[:2000]
                    )
                    answer, provider, _ = model_runtime.generate(
                        system="你是一个文本分类助手，请严格按照指定格式输出分类结果。",
                        prompt=user_prompt,
                        temperature=0,
                    )
                    item.classification = _extract_label(answer)
                    item.reasoning = _extract_reasoning(answer)
                else:
                    # LLM 未配置时，用检索证据做简单匹配 fallback
                    item.classification = "未知（LLM未配置）"
                    item.reasoning = f"检索到 {len(resp.evidence)} 条证据，但 LLM 未启用"
                item.status = "success"
                succeeded += 1

            except Exception as exc:
                item.status = "error"
                item.error = str(exc)[:300]
                failed += 1

            items.append(item)

        return BatchClassifyResponse(
            total_rows=len(rows),
            processed=len(items),
            succeeded=succeeded,
            failed=failed,
            items=items,
            columns=columns,
        )


def _extract_label(answer: str) -> str:
    """从 LLM 回答中提取类别标签。"""
    for label in ["相关·支持", "相关·质疑", "无关·噪音"]:
        if label in answer:
            return label
    # fallback: 取第一行
    first_line = answer.strip().split("\n")[0]
    return first_line[:50] if first_line else "未知"


def _extract_reasoning(answer: str) -> str:
    """从 LLM 回答中提取理由。"""
    if "|" in answer:
        parts = answer.split("|", 1)
        if len(parts) > 1:
            return parts[1].strip()[:300]
    lines = answer.strip().split("\n")
    if len(lines) > 1:
        return "\n".join(lines[1:]).strip()[:300]
    return answer.strip()[:300]
