from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from typing import Any

from ..config import settings
from ..data import TeachingData
from ..models import EvidenceItem, TraceStage
from .document_rag import (
    DenseEncoder,
    QueryTokenizer,
    _dot,
    _group_vector_products,
    _project_vector_3d,
)
from .planner import QueryPlan


PROPERTY_GRAPH_CYPHER = """
MATCH (c:DiseaseCase)-[:observedIn]->(region:Region {id: $region_id})
MATCH (c)-[:hasObservedSymptom]->(symptom:Symptom {id: $symptom_id})
MATCH (c)-[:diagnosedAs]->(disease:Disease)
WHERE c.description STARTS WITH $date_prefix
WITH disease, collect(DISTINCT c) AS cases
RETURN
    disease.id AS disease_id,
    disease.name AS disease_name,
    size(cases) AS case_count
ORDER BY case_count DESC, disease_id
""".strip()


class PropertyGraphRagService:
    def __init__(self, data: TeachingData):
        self.data = data
        self.tokenizer = QueryTokenizer(data)
        self.encoder = DenseEncoder(self.tokenizer)
        self.edges_by_relation = data.edge_lookup

    def _resolve_symptoms(
        self, symptom_text: str, top_k: int = 8
    ) -> tuple[list[dict[str, Any]], list[float]]:
        symptoms = [
            node
            for node in self.data.graph_nodes.values()
            if node["type"] == "Symptom"
        ]
        node_texts = [
            f"{node['name']}。{node['description']}" for node in symptoms
        ]
        vectors = self.encoder.encode_many(node_texts)
        query_vector = self.encoder.encode(symptom_text)
        results = [
            {
                "id": node["id"],
                "name": node["name"],
                "description": node["description"],
                "score": _dot(query_vector, vector),
                "projection": _project_vector_3d(vector),
                "vector": vector,
                "product_groups": _group_vector_products(
                    query_vector,
                    vector,
                    group_count=16,
                ),
            }
            for node, vector in zip(symptoms, vectors, strict=True)
        ]
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k], query_vector

    @staticmethod
    def _date_prefix(plan: QueryPlan) -> str:
        return plan.date_start[:7]

    def _fallback_query(
        self, plan: QueryPlan
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, Any] | None,
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        region_cases = {
            edge["source"]
            for edge in self.edges_by_relation["observedIn"]
            if edge["target"] == plan.region_id
        }
        symptom_cases = {
            edge["source"]
            for edge in self.edges_by_relation["hasObservedSymptom"]
            if edge["target"] == plan.symptom_id
        }
        date_prefix = self._date_prefix(plan)
        matching_cases = {
            case_id
            for case_id in region_cases & symptom_cases
            if self.data.graph_nodes[case_id]["description"].startswith(
                date_prefix
            )
        }

        disease_for_case = {
            edge["source"]: edge["target"]
            for edge in self.edges_by_relation["diagnosedAs"]
        }
        disease_counts = Counter(
            disease_for_case[case_id]
            for case_id in matching_cases
            if case_id in disease_for_case
        )
        disease_rows = [
            {
                "disease_id": disease_id,
                "disease_name": self.data.graph_nodes[disease_id]["name"],
                "case_count": count,
            }
            for disease_id, count in disease_counts.most_common()
        ]
        if not disease_rows:
            return [], [], None, [], {
                "steps": [],
                "matching_case_ids": [],
            }

        main_disease_id = disease_rows[0]["disease_id"]
        target_cases = {
            case_id
            for case_id in matching_cases
            if disease_for_case.get(case_id) == main_disease_id
        }
        symptom_counter: Counter[str] = Counter()
        for edge in self.edges_by_relation["hasObservedSymptom"]:
            if (
                edge["source"] in target_cases
                and edge["target"] != plan.symptom_id
            ):
                symptom_counter[edge["target"]] += 1
        companions = [
            {
                "symptom_id": symptom_id,
                "symptom_name": self.data.graph_nodes[symptom_id]["name"],
                "case_count": count,
            }
            for symptom_id, count in symptom_counter.most_common()
        ]

        recommended = [
            edge
            for edge in self.edges_by_relation["recommendedPesticide"]
            if edge["source"] == main_disease_id
        ]
        pesticide = None
        for edge in recommended:
            node = self.data.graph_nodes[edge["target"]]
            match = re.search(r"安全间隔期(\d+)天", node["description"])
            interval = int(match.group(1)) if match else 999
            if interval <= plan.max_safe_interval_days:
                candidate = {
                    "pesticide_id": node["id"],
                    "pesticide_name": node["name"],
                    "safe_interval_days": interval,
                    "evidence_doc_id": edge["evidence_doc_id"],
                    "description": node["description"],
                }
                if (
                    pesticide is None
                    or interval < pesticide["safe_interval_days"]
                ):
                    pesticide = candidate

        graph_nodes: dict[str, dict[str, str]] = {}
        graph_edges: list[dict[str, str]] = []
        selected_cases = sorted(matching_cases)[:24]
        selected_ids = {
            plan.region_id,
            plan.symptom_id,
            *selected_cases,
        }
        selected_ids.update(disease_counts.keys())
        selected_ids.update(
            row["symptom_id"] for row in companions[:3]
        )
        if pesticide:
            selected_ids.add(pesticide["pesticide_id"])
        for node_id in selected_ids:
            node = self.data.graph_nodes.get(node_id)
            if node:
                graph_nodes[node_id] = node
        for edge in self.data.graph_edges:
            if (
                edge["source"] in graph_nodes
                and edge["target"] in graph_nodes
            ):
                graph_edges.append(edge)

        graph_payload = [
            {
                "id": node["id"],
                "label": node["name"],
                "type": node["type"],
            }
            for node in graph_nodes.values()
        ]
        intersection_cases = region_cases & symptom_cases
        traversal = {
            "anchor_id": plan.symptom_id,
            "matching_case_ids": sorted(matching_cases),
            "main_disease_case_ids": sorted(target_cases),
            "steps": [
                {
                    "index": 1,
                    "operation": "反向沿症状关系找到病例",
                    "relation": "hasObservedSymptom",
                    "input_count": 1,
                    "output_count": len(symptom_cases),
                    "active_ids": sorted(symptom_cases)[:40],
                },
                {
                    "index": 2,
                    "operation": "与地区病例集合求交集",
                    "relation": "observedIn",
                    "input_count": len(symptom_cases),
                    "candidate_count": len(region_cases),
                    "output_count": len(intersection_cases),
                    "active_ids": sorted(intersection_cases)[:40],
                },
                {
                    "index": 3,
                    "operation": "按月份过滤病例属性",
                    "relation": "observedDate",
                    "input_count": len(intersection_cases),
                    "output_count": len(matching_cases),
                    "filter": date_prefix,
                    "active_ids": sorted(matching_cases),
                },
                {
                    "index": 4,
                    "operation": "沿诊断关系扩展并按疾病分组",
                    "relation": "diagnosedAs",
                    "input_count": len(matching_cases),
                    "output_count": len(disease_counts),
                    "groups": disease_rows,
                },
                {
                    "index": 5,
                    "operation": "从主要病害病例扩展伴随症状",
                    "relation": "hasObservedSymptom",
                    "input_count": len(target_cases),
                    "output_count": len(companions),
                    "groups": companions[:8],
                },
            ],
        }
        return disease_rows, companions, pesticide, [
            {"nodes": graph_payload, "edges": graph_edges}
        ], traversal

    def _neo4j_query(
        self, plan: QueryPlan
    ) -> list[dict[str, Any]] | None:
        if not settings.neo4j_password:
            return None
        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
                connection_timeout=2,
            )
            with driver:
                driver.verify_connectivity()
                records, _, _ = driver.execute_query(
                    PROPERTY_GRAPH_CYPHER,
                    region_id=plan.region_id,
                    symptom_id=plan.symptom_id,
                    date_prefix=self._date_prefix(plan),
                    database_=settings.neo4j_database,
                )
                return [record.data() for record in records]
        except Exception:
            return None

    def run(
        self, plan: QueryPlan
    ) -> tuple[str, list[TraceStage], list[EvidenceItem], dict[str, Any], str]:
        started = time.perf_counter()
        anchor_results, anchor_query_vector = self._resolve_symptoms(
            plan.symptom_text
        )
        resolved = anchor_results[0] if anchor_results else None
        if resolved:
            plan.symptom_id = resolved["id"]

        neo4j_rows = self._neo4j_query(plan)
        (
            fallback_rows,
            companions,
            pesticide,
            graph,
            traversal,
        ) = self._fallback_query(plan)
        disease_rows = neo4j_rows or fallback_rows
        source = "Neo4j" if neo4j_rows is not None else "embedded property graph"
        elapsed = int((time.perf_counter() - started) * 1000)
        main = disease_rows[0] if disease_rows else None

        stages = [
            TraceStage(
                id="graph-anchor",
                title="自然语言进入实体向量空间",
                kind="entity-space",
                description=(
                    "问题中的症状描述与每个Symptom节点文本分别编码；"
                    "完整向量内积决定最近节点，三维坐标只用于空间展示。"
                ),
                data={
                    "query": plan.symptom_text,
                    "provider": self.encoder.provider_name,
                    "dimensions": len(anchor_query_vector),
                    "query_vector": [
                        round(value, 6)
                        for value in anchor_query_vector
                    ],
                    "query_projection": _project_vector_3d(
                        anchor_query_vector
                    ),
                    "candidates": [
                        {
                            "id": item["id"],
                            "name": item["name"],
                            "description": item["description"],
                            "score": round(item["score"], 4),
                            "projection": item["projection"],
                            "product_groups": item[
                                "product_groups"
                            ],
                        }
                        for item in anchor_results
                    ],
                    "selected": resolved["id"] if resolved else None,
                    "projection_note": (
                        "展示坐标是稳定降维投影；"
                        "锚点选择使用完整高维向量。"
                    ),
                },
            ),
            TraceStage(
                id="graph-pattern",
                title="把问题约束装配成图模式",
                kind="graph-pattern",
                description=(
                    "先确定允许出现的节点类型、关系方向、过滤条件与跳数，"
                    "Cypher只是这个图模式的可执行文本。"
                ),
                data={
                    "pattern_nodes": [
                        {
                            "id": "case",
                            "label": "DiseaseCase",
                            "role": "candidate",
                        },
                        {
                            "id": "region",
                            "label": "Region",
                            "role": "filter",
                            "value": plan.region_id,
                        },
                        {
                            "id": "symptom",
                            "label": "Symptom",
                            "role": "anchor",
                            "value": plan.symptom_id,
                        },
                        {
                            "id": "disease",
                            "label": "Disease",
                            "role": "group",
                        },
                    ],
                    "pattern_edges": [
                        {
                            "source": "case",
                            "target": "region",
                            "relation": "observedIn",
                            "direction": "out",
                        },
                        {
                            "source": "case",
                            "target": "symptom",
                            "relation": "hasObservedSymptom",
                            "direction": "out",
                        },
                        {
                            "source": "case",
                            "target": "disease",
                            "relation": "diagnosedAs",
                            "direction": "out",
                        },
                    ],
                    "filters": [
                        {
                            "field": "region.id",
                            "operator": "=",
                            "value": plan.region_id,
                        },
                        {
                            "field": "case.observedDate",
                            "operator": "STARTS WITH",
                            "value": self._date_prefix(plan),
                        },
                    ],
                    "allowed_relationships": [
                        "observedIn",
                        "hasObservedSymptom",
                        "diagnosedAs",
                        "recommendedPesticide",
                    ],
                    "blocked_relationships": [
                        "任意标签",
                        "任意关系",
                        "无上限变长路径",
                    ],
                    "max_hops": 3,
                },
            ),
            TraceStage(
                id="graph-traversal",
                title="沿允许关系逐跳扩展与过滤",
                kind="graph-traversal",
                duration_ms=elapsed,
                status=(
                    "completed"
                    if neo4j_rows is not None
                    else "fallback"
                ),
                data={
                    "execution_source": source,
                    "graph": graph[0] if graph else {"nodes": [], "edges": []},
                    "steps": traversal["steps"],
                    "anchor_id": traversal.get("anchor_id"),
                    "matching_case_ids": traversal.get(
                        "matching_case_ids",
                        [],
                    ),
                },
            ),
            TraceStage(
                id="graph-aggregate",
                title="将局部子图压缩为可回答事实",
                kind="graph-aggregate",
                description=(
                    "遍历得到的是节点和边；回答前还要按疾病计数，"
                    "并从主要病害病例继续统计伴随症状。"
                ),
                data={
                    "matched_case_count": sum(
                        row["case_count"] for row in disease_rows
                    ),
                    "disease_counts": disease_rows,
                    "companion_symptoms": companions[:8],
                    "pesticide": pesticide,
                },
            ),
            TraceStage(
                id="cypher-plan",
                title="图模式编译为可复核Cypher",
                kind="code",
                description=(
                    "此代码不是模型任意生成，而是前述模式、白名单关系"
                    "和参数被模板编译后的数据库指令。"
                ),
                data={
                    "language": "cypher",
                    "code": PROPERTY_GRAPH_CYPHER,
                    "parameters": {
                        "region_id": plan.region_id,
                        "symptom_id": plan.symptom_id,
                        "date_prefix": self._date_prefix(plan),
                    },
                    "allowed_path": [
                        "DiseaseCase-observedIn-Region",
                        "DiseaseCase-hasObservedSymptom-Symptom",
                        "DiseaseCase-diagnosedAs-Disease",
                    ],
                    "max_hops": 3,
                },
            ),
        ]

        evidence: list[EvidenceItem] = []
        if pesticide:
            doc = self.data.graph_nodes.get(pesticide["evidence_doc_id"])
            evidence.append(
                EvidenceItem(
                    id=pesticide["evidence_doc_id"],
                    title=doc["name"] if doc else "证据文档",
                    excerpt=pesticide["description"],
                )
            )
        if main:
            matched_count = sum(
                row["case_count"] for row in disease_rows
            )
            companions_text = "、".join(
                f"{row['symptom_name']}{row['case_count']}例"
                for row in companions[:3]
            )
            answer = (
                f"锚点节点为{resolved['name'] if resolved else plan.symptom_id}。"
                f"沿病例—地区—疾病路径扩展后共匹配{matched_count}例，"
                f"{main['disease_name']}以{main['case_count']}例居首；"
                f"伴随症状包括{companions_text}。"
            )
        else:
            answer = "当前图路径没有返回匹配结果。"
        return answer, stages, evidence, {
            "latency_ms": elapsed,
            "anchor_score": (
                round(resolved["score"], 4) if resolved else None
            ),
            "nodes_in_view": (
                len(graph[0]["nodes"]) if graph else 0
            ),
        }, source
