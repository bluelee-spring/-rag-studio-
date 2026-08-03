from __future__ import annotations

import json

from ..models import EvidenceItem, TraceStage
from .model_runtime import model_runtime


class AnswerGenerator:
    """用证据约束LLM；未配置API时保留确定性答案。"""

    @staticmethod
    def _configured() -> bool:
        return model_runtime.answer_enabled

    @staticmethod
    def _stage(
        *,
        strategy: str,
        provider: str,
        question: str,
        deterministic_answer: str,
        final_answer: str,
        evidence: list[EvidenceItem],
        llm_enabled: bool,
        status: str = "completed",
        description: str = "",
    ) -> TraceStage:
        evidence_cards = [
            {
                "id": item.id,
                "title": item.title,
                "excerpt": item.excerpt[:260],
                "source": item.source,
                "score": item.score,
            }
            for item in evidence[:6]
        ]
        return TraceStage(
            id="answer-generation",
            title="证据进入LLM并生成回答",
            kind="generation",
            status=status,
            description=description,
            data={
                "strategy": strategy,
                "provider": provider,
                "llm_enabled": llm_enabled,
                "evidence_count": len(evidence),
                "evidence_ids": [item.id for item in evidence[:12]],
                "evidence_cards": evidence_cards,
                "prompt_layers": [
                    {
                        "role": "system",
                        "label": "回答边界",
                        "content": "只依据检索证据回答，不改变结构化数值。",
                    },
                    {
                        "role": "user",
                        "label": "用户问题",
                        "content": question,
                    },
                    {
                        "role": "context",
                        "label": "结构化结果",
                        "content": deterministic_answer,
                    },
                    {
                        "role": "context",
                        "label": "检索证据",
                        "content": f"{len(evidence_cards)}条证据进入上下文窗口",
                    },
                ],
                "context_characters": (
                    len(question)
                    + len(deterministic_answer)
                    + sum(
                        len(item["excerpt"])
                        for item in evidence_cards
                    )
                ),
                "final_answer": final_answer,
                "output_fragments": [
                    final_answer[index : index + 8]
                    for index in range(0, len(final_answer), 8)
                ],
                "grounding_policy": (
                    "只允许使用检索结果中的事实和数字；"
                    "证据不足时必须明确说明。"
                ),
            },
        )

    def generate(
        self,
        question: str,
        deterministic_answer: str,
        evidence: list[EvidenceItem],
    ) -> tuple[str, TraceStage]:
        if not self._configured():
            return deterministic_answer, self._stage(
                strategy="deterministic-template",
                provider="local",
                question=question,
                deterministic_answer=deterministic_answer,
                final_answer=deterministic_answer,
                evidence=evidence,
                llm_enabled=False,
                status="fallback",
                description=(
                    "当前未配置LLM接口，本轮由可复现模板代替；"
                    "配置API后此处将显示真实模型与上下文生成。"
                ),
            )

        evidence_payload = [
            {
                "id": item.id,
                "title": item.title,
                "excerpt": item.excerpt[:1200],
                "source": item.source,
                "score": item.score,
            }
            for item in evidence[:12]
        ]
        prompt = (
            "请根据给定证据回答问题。保留确定性答案中的所有数值；"
            "不得添加证据外的病害、症状、药剂或安全间隔期。"
            "答案应简洁，并说明主要依据。数据是教学模拟数据。\n"
            f"问题：{question}\n"
            f"确定性结果：{deterministic_answer}\n"
            f"证据：{json.dumps(evidence_payload, ensure_ascii=False)}"
        )

        try:
            answer, provider, _ = model_runtime.generate(
                system=(
                    "你是教学RAG系统的答案生成器。"
                    "只能基于输入证据作答，不得改写数值。"
                ),
                prompt=prompt,
                temperature=0,
            )
            return answer, self._stage(
                strategy="grounded-llm",
                provider=provider,
                question=question,
                deterministic_answer=deterministic_answer,
                final_answer=answer,
                evidence=evidence,
                llm_enabled=True,
                description="模型只接收已检索证据，不直接访问或修改数据库。",
            )
        except Exception as exc:
            stage = self._stage(
                strategy="deterministic-template",
                provider="local-fallback",
                question=question,
                deterministic_answer=deterministic_answer,
                final_answer=deterministic_answer,
                evidence=evidence,
                llm_enabled=False,
                status="warning",
                description=(
                    "生成模型调用失败，本次已回退到确定性结果模板。"
                    f"运行时信息：{str(exc)[:240]}"
                ),
            )
            stage.data["runtime_error"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:500],
            }
            return deterministic_answer, stage
