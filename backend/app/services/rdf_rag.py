from __future__ import annotations

import time
from typing import Any

import httpx
from rdflib import Graph, Namespace, RDF, RDFS, URIRef

from ..config import settings
from ..data import TeachingData
from ..models import EvidenceItem, TraceStage
from .planner import QueryPlan


KG = Namespace("https://example.org/soybean-rag/")
OWL = Namespace("http://www.w3.org/2002/07/owl#")


def disease_count_sparql(plan: QueryPlan) -> str:
    return f"""
PREFIX kg:  <https://example.org/soybean-rag/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?disease ?diseaseName (COUNT(DISTINCT ?case) AS ?caseCount)
WHERE {{
    ?case
        a kg:DiseaseCase ;
        kg:observedIn kg:{plan.region_id} ;
        kg:observedDate ?date ;
        kg:hasObservedSymptom kg:{plan.symptom_id} ;
        kg:diagnosedAs ?disease .

    ?disease kg:name ?diseaseName .

    FILTER(
        ?date >= "{plan.date_start}"^^xsd:date &&
        ?date <  "{plan.date_end}"^^xsd:date
    )
}}
GROUP BY ?disease ?diseaseName
ORDER BY DESC(?caseCount) ?disease
""".strip()


def companion_sparql(plan: QueryPlan, disease_id: str) -> str:
    return f"""
PREFIX kg:  <https://example.org/soybean-rag/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?symptom ?symptomName (COUNT(DISTINCT ?case) AS ?caseCount)
WHERE {{
    ?case
        a kg:DiseaseCase ;
        kg:observedIn kg:{plan.region_id} ;
        kg:observedDate ?date ;
        kg:hasObservedSymptom kg:{plan.symptom_id} ;
        kg:hasObservedSymptom ?symptom ;
        kg:diagnosedAs kg:{disease_id} .

    ?symptom kg:name ?symptomName .

    FILTER(?symptom != kg:{plan.symptom_id})
    FILTER(
        ?date >= "{plan.date_start}"^^xsd:date &&
        ?date <  "{plan.date_end}"^^xsd:date
    )
}}
GROUP BY ?symptom ?symptomName
ORDER BY DESC(?caseCount) ?symptom
""".strip()


def pesticide_sparql(plan: QueryPlan, disease_id: str) -> str:
    return f"""
PREFIX kg: <https://example.org/soybean-rag/>

SELECT
    ?pesticide
    ?pesticideName
    ?safeIntervalDays
    ?document
    ?documentName
WHERE {{
    kg:{disease_id}
        kg:recommendedPesticide
        ?pesticide .

    ?pesticide
        kg:name ?pesticideName ;
        kg:safeIntervalDays ?safeIntervalDays ;
        kg:evidenceFrom ?document .

    ?document kg:name ?documentName .

    FILTER(?safeIntervalDays <= {plan.max_safe_interval_days})
}}
ORDER BY ?safeIntervalDays ?pesticide
LIMIT 1
""".strip()


class RdfRagService:
    def __init__(self, data: TeachingData):
        self.data = data
        self.graph = Graph()
        self.graph.parse(data.ontology_path, format="turtle")
        self.graph.parse(data.instances_path, format="turtle")
        self.shapes = Graph()
        self.shapes.parse(data.shapes_path, format="turtle")

    @staticmethod
    def _local_name(value: Any) -> str:
        text = str(value)
        return text.rsplit("/", 1)[-1].rsplit("#", 1)[-1]

    def _local_select(self, query: str) -> list[dict[str, Any]]:
        result = self.graph.query(query)
        rows: list[dict[str, Any]] = []
        for binding in result:
            payload: dict[str, Any] = {}
            for variable, value in binding.asdict().items():
                payload[str(variable)] = (
                    int(value)
                    if getattr(value, "datatype", None)
                    and str(value.datatype).endswith(
                        ("#integer", "#int")
                    )
                    else str(value)
                )
            rows.append(payload)
        return rows

    def _fuseki_select(self, query: str) -> list[dict[str, Any]]:
        response = httpx.post(
            settings.fuseki_query_url,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        rows: list[dict[str, Any]] = []
        for binding in payload["results"]["bindings"]:
            row: dict[str, Any] = {}
            for key, item in binding.items():
                value = item["value"]
                if item.get("datatype", "").endswith(
                    ("#integer", "#int")
                ):
                    value = int(value)
                row[key] = value
            rows.append(row)
        return rows

    def _select(self, query: str) -> tuple[list[dict[str, Any]], str]:
        if settings.fuseki_query_url:
            try:
                return self._fuseki_select(query), "Apache Jena Fuseki"
            except Exception:
                pass
        return self._local_select(query), "embedded RDFLib"

    def _ontology_summary(self) -> dict[str, Any]:
        classes = [
            {
                "id": self._local_name(subject),
                "label": str(label) if label else self._local_name(subject),
            }
            for subject in self.graph.subjects(RDF.type, OWL.Class)
            for label in [self.graph.value(subject, RDFS.label)]
        ]
        object_properties = [
            {
                "id": self._local_name(subject),
                "domain": self._local_name(
                    self.graph.value(subject, RDFS.domain)
                )
                if self.graph.value(subject, RDFS.domain)
                else "Entity",
                "range": self._local_name(
                    self.graph.value(subject, RDFS.range)
                )
                if self.graph.value(subject, RDFS.range)
                else "Document",
            }
            for subject in self.graph.subjects(
                RDF.type, OWL.ObjectProperty
            )
        ]
        datatype_properties = [
            self._local_name(subject)
            for subject in self.graph.subjects(
                RDF.type, OWL.DatatypeProperty
            )
        ]
        return {
            "classes": classes,
            "object_properties": object_properties,
            "datatype_properties": datatype_properties,
            "shape_triples": len(self.shapes),
        }

    def _pesticide_filter_trace(
        self,
        disease_id: str,
        max_safe_interval_days: int,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        disease_iri = KG[disease_id]
        for pesticide_iri in self.graph.objects(
            disease_iri,
            KG.recommendedPesticide,
        ):
            interval_value = self.graph.value(
                pesticide_iri,
                KG.safeIntervalDays,
            )
            interval = (
                int(interval_value)
                if interval_value is not None
                else 999
            )
            evidence_iri = self.graph.value(
                pesticide_iri,
                KG.evidenceFrom,
            )
            candidates.append(
                {
                    "id": self._local_name(pesticide_iri),
                    "name": str(
                        self.graph.value(pesticide_iri, KG.name)
                        or self._local_name(pesticide_iri)
                    ),
                    "safe_interval_days": interval,
                    "operator": "<=",
                    "threshold": max_safe_interval_days,
                    "passed": interval <= max_safe_interval_days,
                    "evidence_id": (
                        self._local_name(evidence_iri)
                        if evidence_iri
                        else None
                    ),
                }
            )
        candidates.sort(
            key=lambda item: (
                not item["passed"],
                item["safe_interval_days"],
                item["id"],
            )
        )
        return candidates

    def _evidence_graph(
        self,
        plan: QueryPlan,
        disease_id: str,
        companions: list[dict[str, Any]],
        pesticide: dict[str, Any] | None,
    ) -> dict[str, Any]:
        ids = {
            plan.region_id,
            plan.symptom_id,
            disease_id,
        }
        ids.update(
            self._local_name(row["symptom"])
            for row in companions[:3]
        )
        if pesticide:
            ids.add(self._local_name(pesticide["pesticide"]))
            ids.add(self._local_name(pesticide["document"]))

        nodes = []
        for item_id in ids:
            iri = KG[item_id]
            name = self.graph.value(iri, KG.name)
            node_type = self.graph.value(iri, RDF.type)
            nodes.append(
                {
                    "id": item_id,
                    "label": str(name) if name else item_id,
                    "type": (
                        self._local_name(node_type)
                        if node_type
                        else "Resource"
                    ),
                }
            )

        edges = []
        allowed_predicates = {
            KG.observedIn,
            KG.hasObservedSymptom,
            KG.diagnosedAs,
            KG.recommendedPesticide,
            KG.evidenceFrom,
        }
        for subject, predicate, obj in self.graph:
            source_id = self._local_name(subject)
            target_id = self._local_name(obj)
            if (
                predicate in allowed_predicates
                and source_id in ids
                and target_id in ids
            ):
                edges.append(
                    {
                        "source": source_id,
                        "target": target_id,
                        "relation": self._local_name(predicate),
                    }
                )
        return {"nodes": nodes, "edges": edges}

    def run(
        self, plan: QueryPlan
    ) -> tuple[str, list[TraceStage], list[EvidenceItem], dict[str, Any], str]:
        started = time.perf_counter()
        count_query = disease_count_sparql(plan)
        disease_rows, source = self._select(count_query)
        normalized_diseases = [
            {
                "disease_id": self._local_name(row["disease"]),
                "disease_name": row["diseaseName"],
                "case_count": int(row["caseCount"]),
            }
            for row in disease_rows
        ]
        main = (
            normalized_diseases[0]
            if normalized_diseases
            else {
                "disease_id": plan.disease_id or "DIS-01",
                "disease_name": "未命中",
                "case_count": 0,
            }
        )

        companion_query = companion_sparql(
            plan, main["disease_id"]
        )
        companion_rows, _ = self._select(companion_query)
        companions = [
            {
                "symptom": row["symptom"],
                "symptom_id": self._local_name(row["symptom"]),
                "symptom_name": row["symptomName"],
                "case_count": int(row["caseCount"]),
            }
            for row in companion_rows
        ]

        pesticide_query = pesticide_sparql(
            plan, main["disease_id"]
        )
        pesticide_rows, _ = self._select(pesticide_query)
        pesticide = pesticide_rows[0] if pesticide_rows else None
        if pesticide:
            pesticide = {
                **pesticide,
                "pesticide_id": self._local_name(
                    pesticide["pesticide"]
                ),
                "document_id": self._local_name(
                    pesticide["document"]
                ),
                "safeIntervalDays": int(
                    pesticide["safeIntervalDays"]
                ),
            }

        graph = self._evidence_graph(
            plan,
            main["disease_id"],
            companions,
            pesticide,
        )
        pesticide_candidates = self._pesticide_filter_trace(
            main["disease_id"],
            plan.max_safe_interval_days,
        )
        elapsed = int((time.perf_counter() - started) * 1000)

        stages = [
            TraceStage(
                id="rdf-ontology",
                title="本体规定允许出现的语义结构",
                kind="ontology-space",
                description=(
                    "类定义节点类型，对象属性规定节点间可连接的谓语，"
                    "数据属性规定资源可以携带的数值。"
                ),
                data=self._ontology_summary(),
            ),
            TraceStage(
                id="rdf-plan",
                title="自然语言实体映射为全局IRI",
                kind="iri-mapping",
                description=(
                    "名称只是界面文本；RDF查询实际使用唯一IRI，"
                    "从而避免同名实体和跨系统标识冲突。"
                ),
                data={
                    "mappings": [
                        {
                            "source": "六合区",
                            "role": "地区实体",
                            "local_id": plan.region_id,
                            "iri": str(KG[plan.region_id]),
                        },
                        {
                            "source": plan.symptom_text,
                            "role": "症状实体",
                            "local_id": plan.symptom_id,
                            "iri": str(KG[plan.symptom_id]),
                        },
                        {
                            "source": main["disease_name"],
                            "role": "疾病实体",
                            "local_id": main["disease_id"],
                            "iri": str(KG[main["disease_id"]]),
                        },
                    ],
                    "region_iri": str(KG[plan.region_id]),
                    "symptom_iri": str(KG[plan.symptom_id]),
                    "date_range": [
                        plan.date_start,
                        plan.date_end,
                    ],
                    "max_safe_interval_days": (
                        plan.max_safe_interval_days
                    ),
                    "allowed_predicates": [
                        "observedIn",
                        "observedDate",
                        "hasObservedSymptom",
                        "diagnosedAs",
                        "recommendedPesticide",
                        "safeIntervalDays",
                        "evidenceFrom",
                    ],
                },
            ),
            TraceStage(
                id="rdf-pattern",
                title="变量与三元组组成图模式",
                kind="triple-pattern",
                description=(
                    "SPARQL不是在全文中找字符串，而是寻找同时满足"
                    "这些主语—谓语—宾语结构的变量绑定。"
                ),
                data={
                    "nodes": [
                        {"id": "case", "label": "?case", "kind": "variable"},
                        {
                            "id": "region",
                            "label": plan.region_id,
                            "kind": "iri",
                        },
                        {
                            "id": "symptom",
                            "label": plan.symptom_id,
                            "kind": "iri",
                        },
                        {
                            "id": "disease",
                            "label": "?disease",
                            "kind": "variable",
                        },
                        {
                            "id": "date",
                            "label": "?date",
                            "kind": "literal",
                        },
                    ],
                    "patterns": [
                        {
                            "subject": "case",
                            "predicate": "rdf:type",
                            "object": "DiseaseCase",
                        },
                        {
                            "subject": "case",
                            "predicate": "observedIn",
                            "object": "region",
                        },
                        {
                            "subject": "case",
                            "predicate": "hasObservedSymptom",
                            "object": "symptom",
                        },
                        {
                            "subject": "case",
                            "predicate": "diagnosedAs",
                            "object": "disease",
                        },
                        {
                            "subject": "case",
                            "predicate": "observedDate",
                            "object": "date",
                        },
                    ],
                    "filters": [
                        {
                            "variable": "?date",
                            "operator": ">=",
                            "value": plan.date_start,
                        },
                        {
                            "variable": "?date",
                            "operator": "<",
                            "value": plan.date_end,
                        },
                    ],
                },
            ),
            TraceStage(
                id="rdf-filter",
                title="数据属性通过数值约束筛选",
                kind="rdf-filter",
                description=(
                    "每个药剂资源携带safeIntervalDays字面量；"
                    "过滤器比较真实数值，只保留小于等于阈值的资源。"
                ),
                data={
                    "property": "safeIntervalDays",
                    "operator": "<=",
                    "threshold": plan.max_safe_interval_days,
                    "candidates": pesticide_candidates,
                },
            ),
            TraceStage(
                id="rdf-subgraph",
                title="变量绑定回收为证据三元组",
                kind="graph-traversal",
                duration_ms=elapsed,
                status=(
                    "completed"
                    if source == "Apache Jena Fuseki"
                    else "fallback"
                ),
                data={
                    "execution_source": source,
                    "graph": graph,
                    "steps": [
                        {
                            "index": 1,
                            "operation": "匹配病例图模式",
                            "output_count": sum(
                                row["case_count"]
                                for row in normalized_diseases
                            ),
                        },
                        {
                            "index": 2,
                            "operation": "绑定疾病变量并分组",
                            "output_count": len(
                                normalized_diseases
                            ),
                        },
                        {
                            "index": 3,
                            "operation": "扩展伴随症状",
                            "output_count": len(companions),
                        },
                        {
                            "index": 4,
                            "operation": "回收药剂及证据来源",
                            "output_count": 1 if pesticide else 0,
                        },
                    ],
                    "disease_counts": normalized_diseases,
                    "companion_symptoms": companions[:8],
                    "pesticide": pesticide,
                },
            ),
            TraceStage(
                id="sparql-generate",
                title="图模式编译为可复核SPARQL",
                kind="code",
                data={
                    "language": "sparql",
                    "code": count_query,
                    "follow_up_queries": [
                        {
                            "title": "伴随症状",
                            "code": companion_query,
                        },
                        {
                            "title": "药剂与证据",
                            "code": pesticide_query,
                        },
                    ],
                },
            ),
        ]

        evidence: list[EvidenceItem] = []
        if pesticide:
            evidence.append(
                EvidenceItem(
                    id=pesticide["document_id"],
                    title=pesticide["documentName"],
                    excerpt=(
                        f"{pesticide['pesticideName']}；"
                        f"模拟安全间隔期"
                        f"{pesticide['safeIntervalDays']}天。"
                    ),
                )
            )

        matched_count = sum(
            row["case_count"] for row in normalized_diseases
        )
        companions_text = "、".join(
            f"{row['symptom_name']}{row['case_count']}例"
            for row in companions[:3]
        )
        pesticide_text = (
            f"{pesticide['pesticideName']}，模拟安全间隔期"
            f"{pesticide['safeIntervalDays']}天"
            if pesticide
            else "没有符合条件的药剂资源"
        )
        answer = (
            f"SPARQL在当前RDF数据集中匹配{matched_count}例；"
            f"{main['disease_name']}{main['case_count']}例。"
            f"伴随症状包括{companions_text}；"
            f"药剂筛选结果为{pesticide_text}。"
        )
        return answer, stages, evidence, {
            "latency_ms": elapsed,
            "rdf_triples": len(self.graph),
            "matched_cases": matched_count,
        }, source
