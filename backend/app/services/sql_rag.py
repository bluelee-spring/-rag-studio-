from __future__ import annotations

import time
from typing import Any

from ..data import TeachingData
from ..models import EvidenceItem, TraceStage
from .planner import QueryPlan


DISEASE_COUNT_SQL = """
WITH matched_cases AS (
    SELECT DISTINCT fc.case_id, fc.disease_id
    FROM field_case AS fc
    JOIN case_symptom AS cs
      ON cs.case_id = fc.case_id
    WHERE fc.region_id = :region_id
      AND fc.observed_date >= :date_start
      AND fc.observed_date <  :date_end
      AND cs.symptom_id = :symptom_id
)
SELECT
    d.disease_id,
    d.name AS disease_name,
    COUNT(*) AS case_count
FROM matched_cases AS mc
JOIN disease AS d
  ON d.disease_id = mc.disease_id
GROUP BY d.disease_id, d.name
ORDER BY case_count DESC, d.disease_id
""".strip()


COMPANION_SQL = """
WITH target_cases AS (
    SELECT DISTINCT fc.case_id
    FROM field_case AS fc
    JOIN case_symptom AS anchor
      ON anchor.case_id = fc.case_id
    WHERE fc.region_id = :region_id
      AND fc.observed_date >= :date_start
      AND fc.observed_date <  :date_end
      AND anchor.symptom_id = :symptom_id
      AND fc.disease_id = :disease_id
)
SELECT
    s.symptom_id,
    s.name AS symptom_name,
    COUNT(DISTINCT tc.case_id) AS case_count
FROM target_cases AS tc
JOIN case_symptom AS cs
  ON cs.case_id = tc.case_id
JOIN symptom AS s
  ON s.symptom_id = cs.symptom_id
WHERE s.symptom_id <> :symptom_id
GROUP BY s.symptom_id, s.name
ORDER BY case_count DESC, s.symptom_id
""".strip()


PESTICIDE_SQL = """
SELECT
    p.pesticide_id,
    p.name AS pesticide_name,
    dp.safe_interval_days,
    dp.dosage_text,
    dp.source_doc_id,
    doc.title AS evidence_title,
    doc.content AS evidence_content
FROM disease_pesticide AS dp
JOIN pesticide AS p
  ON p.pesticide_id = dp.pesticide_id
JOIN document AS doc
  ON doc.doc_id = dp.source_doc_id
WHERE dp.disease_id = :disease_id
  AND dp.safe_interval_days <= :max_safe_interval_days
ORDER BY dp.recommendation_priority, dp.safe_interval_days
LIMIT 1
""".strip()


class SqlRagService:
    def __init__(self, data: TeachingData):
        self.data = data

    def run(
        self, plan: QueryPlan
    ) -> tuple[str, list[TraceStage], list[EvidenceItem], dict[str, Any]]:
        started = time.perf_counter()
        params: dict[str, Any] = {
            "region_id": plan.region_id,
            "date_start": plan.date_start,
            "date_end": plan.date_end,
            "symptom_id": plan.symptom_id,
            "max_safe_interval_days": plan.max_safe_interval_days,
        }

        with self.data.sqlite() as connection:
            filter_counts = {
                "all_cases": connection.execute(
                    "SELECT COUNT(*) FROM field_case"
                ).fetchone()[0],
                "region_cases": connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM field_case
                    WHERE region_id = :region_id
                    """,
                    params,
                ).fetchone()[0],
                "region_date_cases": connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM field_case
                    WHERE region_id = :region_id
                      AND observed_date >= :date_start
                      AND observed_date < :date_end
                    """,
                    params,
                ).fetchone()[0],
                "joined_symptom_cases": connection.execute(
                    """
                    SELECT COUNT(DISTINCT fc.case_id)
                    FROM field_case AS fc
                    JOIN case_symptom AS cs
                      ON cs.case_id = fc.case_id
                    WHERE fc.region_id = :region_id
                      AND fc.observed_date >= :date_start
                      AND fc.observed_date < :date_end
                      AND cs.symptom_id = :symptom_id
                    """,
                    params,
                ).fetchone()[0],
            }
            joined_samples = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT
                        fc.case_id,
                        fc.region_id,
                        fc.observed_date,
                        cs.symptom_id,
                        fc.disease_id,
                        d.name AS disease_name
                    FROM field_case AS fc
                    JOIN case_symptom AS cs
                      ON cs.case_id = fc.case_id
                    LEFT JOIN disease AS d
                      ON d.disease_id = fc.disease_id
                    WHERE fc.region_id = :region_id
                      AND fc.observed_date >= :date_start
                      AND fc.observed_date < :date_end
                      AND cs.symptom_id = :symptom_id
                    ORDER BY fc.case_id
                    LIMIT 8
                    """,
                    params,
                ).fetchall()
            ]
            explain_rows = connection.execute(
                f"EXPLAIN QUERY PLAN {DISEASE_COUNT_SQL}",
                params,
            ).fetchall()
            disease_rows = [
                dict(row)
                for row in connection.execute(
                    DISEASE_COUNT_SQL,
                    params,
                ).fetchall()
            ]

            main_disease = (
                disease_rows[0]
                if disease_rows
                else {
                    "disease_id": plan.disease_id or "DIS-01",
                    "disease_name": "未命中",
                    "case_count": 0,
                }
            )
            params["disease_id"] = main_disease["disease_id"]

            companion_rows = [
                dict(row)
                for row in connection.execute(
                    COMPANION_SQL,
                    params,
                ).fetchall()
            ]
            pesticide_row = connection.execute(
                PESTICIDE_SQL,
                params,
            ).fetchone()
            pesticide = dict(pesticide_row) if pesticide_row else None

        matched_count = sum(row["case_count"] for row in disease_rows)
        elapsed = int((time.perf_counter() - started) * 1000)

        stages = [
            TraceStage(
                id="sql-plan",
                title="自然语言约束映射为字段条件",
                kind="relational-plan",
                data={
                    "planner": plan.planner,
                    "plan": plan.to_dict(),
                    "allowed_tables": [
                        "field_case",
                        "case_symptom",
                        "disease",
                        "symptom",
                        "disease_pesticide",
                        "pesticide",
                        "document",
                    ],
                },
            ),
            TraceStage(
                id="sql-filter",
                title="数据行依次通过选择条件",
                kind="row-filter",
                description=(
                    "关系数据库先在field_case中缩小行集合，"
                    "再通过case_id连接症状表，得到满足全部条件的病例。"
                ),
                data={
                    "filters": [
                        {
                            "label": "全部病例",
                            "field": "*",
                            "operator": "SCAN",
                            "value": "field_case",
                            "count": filter_counts["all_cases"],
                        },
                        {
                            "label": "地区选择",
                            "field": "region_id",
                            "operator": "=",
                            "value": plan.region_id,
                            "count": filter_counts["region_cases"],
                        },
                        {
                            "label": "时间范围",
                            "field": "observed_date",
                            "operator": "[start, end)",
                            "value": (
                                f"{plan.date_start} → {plan.date_end}"
                            ),
                            "count": filter_counts[
                                "region_date_cases"
                            ],
                        },
                        {
                            "label": "症状连接",
                            "field": "case_symptom.symptom_id",
                            "operator": "=",
                            "value": plan.symptom_id,
                            "count": filter_counts[
                                "joined_symptom_cases"
                            ],
                        },
                    ],
                },
            ),
            TraceStage(
                id="sql-join",
                title="主键与外键把表外数据接回同一行",
                kind="key-join",
                description=(
                    "field_case.case_id与case_symptom.case_id值相等，"
                    "数据库据此组合两张表；disease_id再连接疾病名称。"
                ),
                data={
                    "tables": [
                        {
                            "name": "field_case",
                            "primary_key": "case_id",
                            "columns": [
                                "case_id",
                                "region_id",
                                "observed_date",
                                "disease_id",
                            ],
                        },
                        {
                            "name": "case_symptom",
                            "primary_key": (
                                "case_id + symptom_id"
                            ),
                            "foreign_keys": [
                                {
                                    "column": "case_id",
                                    "references": (
                                        "field_case.case_id"
                                    ),
                                }
                            ],
                            "columns": [
                                "case_id",
                                "symptom_id",
                            ],
                        },
                        {
                            "name": "disease",
                            "primary_key": "disease_id",
                            "columns": [
                                "disease_id",
                                "name",
                            ],
                        },
                    ],
                    "joins": [
                        {
                            "left": "field_case.case_id",
                            "right": "case_symptom.case_id",
                            "type": "INNER JOIN",
                        },
                        {
                            "left": "field_case.disease_id",
                            "right": "disease.disease_id",
                            "type": "INNER JOIN",
                        },
                    ],
                    "sample_rows": joined_samples,
                },
            ),
            TraceStage(
                id="sql-result",
                title="按疾病分组并执行COUNT",
                kind="group-aggregate",
                duration_ms=elapsed,
                data={
                    "matched_case_count": matched_count,
                    "disease_counts": disease_rows,
                    "main_disease": main_disease,
                    "companion_symptoms": companion_rows[:8],
                    "pesticide": pesticide,
                },
            ),
            TraceStage(
                id="sql-generate",
                title="关系运算编译为参数化SQL",
                kind="code",
                description=(
                    "查询结构由模板确定，用户值只通过命名参数进入执行器。"
                ),
                data={
                    "language": "sql",
                    "code": DISEASE_COUNT_SQL,
                    "parameters": {
                        key: value
                        for key, value in params.items()
                        if key != "disease_id"
                    },
                },
            ),
            TraceStage(
                id="sql-explain",
                title="SQLite实际执行计划",
                kind="query-plan",
                data={
                    "rows": [
                        {
                            "id": row["id"],
                            "parent": row["parent"],
                            "detail": row["detail"],
                        }
                        for row in explain_rows
                    ]
                },
            ),
        ]

        evidence: list[EvidenceItem] = []
        if pesticide:
            evidence.append(
                EvidenceItem(
                    id=pesticide["source_doc_id"],
                    title=pesticide["evidence_title"],
                    excerpt=pesticide["evidence_content"][:300],
                )
            )

        companions = "、".join(
            f"{row['symptom_name']}{row['case_count']}例"
            for row in companion_rows[:3]
        )
        pesticide_text = (
            f"{pesticide['pesticide_name']}，模拟安全间隔期"
            f"{pesticide['safe_interval_days']}天"
            if pesticide
            else "未找到符合条件的教学药剂"
        )
        answer = (
            f"共匹配{matched_count}例；其中"
            f"{main_disease['disease_name']}{main_disease['case_count']}例，"
            f"为频率最高的候选病害。常见伴随症状为{companions}。"
            f"推荐结果：{pesticide_text}。"
        )
        return answer, stages, evidence, {
            "latency_ms": elapsed,
            "matched_cases": matched_count,
            "main_disease_cases": main_disease["case_count"],
            "query_engine": "SQLite",
        }
