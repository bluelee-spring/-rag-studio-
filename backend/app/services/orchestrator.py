from __future__ import annotations

import time
import uuid

from ..data import TeachingData
from ..models import (
    EvidenceItem,
    QueryRequest,
    QueryResponse,
    TraceStage,
)
from .answer_generator import AnswerGenerator
from .document_rag import DocumentRagService
from .planner import QueryPlanner
from .property_graph_rag import PropertyGraphRagService
from .rdf_rag import RdfRagService
from .sql_rag import SqlRagService


MODE_NAMES = {
    "tfidf": "TF–IDF",
    "bm25": "BM25",
    "semantic": "语义向量",
    "property_graph": "属性图",
    "rdf": "RDF / SPARQL",
    "sql": "关系数据库",
    "composite": "综合路由",
}


class RagOrchestrator:
    def __init__(self, data: TeachingData):
        self.data = data
        self.planner = QueryPlanner(data)
        self.documents = DocumentRagService(data)
        self.property_graph = PropertyGraphRagService(data)
        self.rdf = RdfRagService(data)
        self.sql = SqlRagService(data)
        self.answer_generator = AnswerGenerator()

    @staticmethod
    def _dedupe_evidence(
        evidence: list[EvidenceItem],
    ) -> list[EvidenceItem]:
        seen: set[str] = set()
        result: list[EvidenceItem] = []
        for item in evidence:
            if item.id in seen:
                continue
            seen.add(item.id)
            result.append(item)
        return result

    def _composite(
        self,
        request: QueryRequest,
    ) -> tuple[
        str,
        list[TraceStage],
        list[EvidenceItem],
        dict,
        str,
    ]:
        started = time.perf_counter()
        plan = self.planner.plan(request.question)

        _, graph_stages, graph_evidence, graph_metrics, graph_source = (
            self.property_graph.run(plan)
        )
        sql_answer, sql_stages, sql_evidence, sql_metrics = self.sql.run(
            plan
        )
        _, rdf_stages, rdf_evidence, rdf_metrics, rdf_source = self.rdf.run(
            plan
        )

        stages = [
            TraceStage(
                id="route-plan",
                title="任务分解",
                kind="route",
                data={
                    "planner": plan.planner,
                    "question": request.question,
                    "plan": plan.to_dict(),
                    "routes": [
                        {
                            "action": "症状实体对齐",
                            "executor": "语义向量",
                        },
                        {
                            "action": "地区、日期与病例统计",
                            "executor": "SQLite",
                        },
                        {
                            "action": "伴随症状局部扩展",
                            "executor": "属性图",
                        },
                        {
                            "action": "药剂约束与证据",
                            "executor": "RDF / SPARQL",
                        },
                    ],
                },
            ),
            graph_stages[0],
            sql_stages[1],
            sql_stages[2],
            sql_stages[3],
            graph_stages[2],
            graph_stages[3],
            rdf_stages[3],
            rdf_stages[4],
            TraceStage(
                id="context-assembly",
                title="证据合并",
                kind="context",
                data={
                    "policy": (
                        "计数采用关系数据库结果；实体邻域采用属性图；"
                        "药剂数值约束和IRI证据采用RDF。"
                    ),
                    "conflict_strategy": (
                        "结构化数据库结果优先，文本检索仅提供补充证据。"
                    ),
                },
            ),
        ]
        evidence = self._dedupe_evidence(
            graph_evidence + sql_evidence + rdf_evidence
        )
        elapsed = int((time.perf_counter() - started) * 1000)
        metrics = {
            "latency_ms": elapsed,
            "subtasks": 5,
            "matched_cases": sql_metrics.get("matched_cases"),
            "graph_anchor_score": graph_metrics.get("anchor_score"),
            "rdf_triples": rdf_metrics.get("rdf_triples"),
        }
        return (
            sql_answer,
            stages,
            evidence,
            metrics,
            f"SQLite + {graph_source} + {rdf_source}",
        )

    def run(self, request: QueryRequest) -> QueryResponse:
        plan = self.planner.plan(request.question)
        source = "embedded"

        if request.mode == "tfidf":
            answer, stages, evidence, metrics = self.documents.tfidf(
                request.question,
                request.top_k,
            )
            source = "document corpus"
        elif request.mode == "bm25":
            answer, stages, evidence, metrics = self.documents.bm25(
                request.question,
                request.top_k,
            )
            source = "document corpus"
        elif request.mode == "semantic":
            answer, stages, evidence, metrics = self.documents.semantic(
                request.question,
                request.top_k,
            )
            source = metrics["provider"]
        elif request.mode == "property_graph":
            (
                answer,
                stages,
                evidence,
                metrics,
                source,
            ) = self.property_graph.run(plan)
        elif request.mode == "rdf":
            (
                answer,
                stages,
                evidence,
                metrics,
                source,
            ) = self.rdf.run(plan)
        elif request.mode == "sql":
            answer, stages, evidence, metrics = self.sql.run(plan)
            source = "SQLite"
        else:
            (
                answer,
                stages,
                evidence,
                metrics,
                source,
            ) = self._composite(request)

        answer, generation_stage = self.answer_generator.generate(
            request.question,
            answer,
            evidence,
        )
        stages.append(generation_stage)

        return QueryResponse(
            run_id=f"run_{uuid.uuid4().hex[:12]}",
            mode=request.mode,
            mode_name=MODE_NAMES[request.mode],
            question=request.question,
            answer=answer,
            stages=stages,
            evidence=evidence,
            metrics=metrics,
            execution_source=source,
        )
