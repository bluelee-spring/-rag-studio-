from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from ..data import TeachingData
from .model_runtime import model_runtime


@dataclass
class QueryPlan:
    region_id: str = "REG-01"
    region_name: str = "六合区"
    symptom_id: str = "SYM-01"
    symptom_text: str = "褐色近圆形病斑"
    date_start: str = "2025-07-01"
    date_end: str = "2025-08-01"
    max_safe_interval_days: int = 14
    disease_id: str | None = None
    actions: tuple[str, ...] = (
        "RESOLVE_SYMPTOM",
        "COUNT_DISEASES",
        "FIND_COMPANION_SYMPTOMS",
        "FIND_PESTICIDE",
        "RETRIEVE_EVIDENCE",
    )
    planner: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["actions"] = list(self.actions)
        return payload


class QueryPlanner:
    def __init__(self, data: TeachingData):
        self.data = data
        self.regions = self._load_regions()
        self.symptoms = self._load_symptoms()
        self.diseases = self._load_diseases()

    def _load_regions(self) -> list[dict[str, str]]:
        with self.data.sqlite() as connection:
            rows = connection.execute(
                "SELECT region_id, name FROM region"
            ).fetchall()
        return [dict(row) for row in rows]

    def _load_symptoms(self) -> list[dict[str, str]]:
        with self.data.sqlite() as connection:
            rows = connection.execute(
                "SELECT symptom_id, name, aliases FROM symptom"
            ).fetchall()
        return [dict(row) for row in rows]

    def _load_diseases(self) -> list[dict[str, str]]:
        with self.data.sqlite() as connection:
            rows = connection.execute(
                "SELECT disease_id, name, aliases FROM disease"
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _first_match(
        question: str,
        rows: list[dict[str, str]],
        id_key: str,
    ) -> dict[str, str] | None:
        for row in rows:
            candidates = [row.get("name", "")]
            candidates.extend(
                part.strip()
                for part in (row.get("aliases") or "").split("|")
                if part.strip()
            )
            if any(candidate and candidate in question for candidate in candidates):
                return row
        return None

    def deterministic_plan(self, question: str) -> QueryPlan:
        plan = QueryPlan()

        region = self._first_match(question, self.regions, "region_id")
        if region:
            plan.region_id = region["region_id"]
            plan.region_name = region["name"]

        symptom = self._first_match(question, self.symptoms, "symptom_id")
        if symptom:
            plan.symptom_id = symptom["symptom_id"]
            plan.symptom_text = symptom["name"]

        disease = self._first_match(question, self.diseases, "disease_id")
        if disease:
            plan.disease_id = disease["disease_id"]

        year_match = re.search(r"(20\d{2})年", question)
        month_match = re.search(r"(1[0-2]|0?[1-9])月", question)
        if year_match and month_match:
            year = int(year_match.group(1))
            month = int(month_match.group(1))
            next_year = year + (1 if month == 12 else 0)
            next_month = 1 if month == 12 else month + 1
            plan.date_start = f"{year:04d}-{month:02d}-01"
            plan.date_end = f"{next_year:04d}-{next_month:02d}-01"

        interval_match = re.search(
            r"(?:不超过|小于等于|≤)\s*(\d+)\s*天",
            question,
        )
        if interval_match:
            plan.max_safe_interval_days = int(interval_match.group(1))

        return plan

    def _llm_plan(self, question: str, fallback: QueryPlan) -> QueryPlan:
        if not model_runtime.planner_enabled:
            return fallback

        schema = {
            "regions": self.regions,
            "symptoms": self.symptoms,
            "diseases": self.diseases,
            "allowed_actions": list(fallback.actions),
        }
        prompt = (
            "将用户问题转换成受约束查询计划。只能使用给定ID和action，"
            "只输出JSON。缺失字段沿用default_plan。\n"
            f"schema={json.dumps(schema, ensure_ascii=False)}\n"
            f"default_plan={json.dumps(fallback.to_dict(), ensure_ascii=False)}\n"
            f"question={question}"
        )

        try:
            content, _, _ = model_runtime.generate(
                system=(
                    "你是数据库查询计划器。禁止生成SQL、Cypher、"
                    "SPARQL，只能生成受约束JSON计划。"
                ),
                prompt=prompt,
                temperature=0,
                max_tokens=600,
            )
            cleaned = re.sub(
                r"^```(?:json)?\s*|\s*```$", "", content.strip()
            )
            payload = json.loads(cleaned)
            allowed = set(fallback.actions)
            actions = tuple(
                action
                for action in payload.get("actions", fallback.actions)
                if action in allowed
            )
            return QueryPlan(
                region_id=payload.get("region_id", fallback.region_id),
                region_name=payload.get("region_name", fallback.region_name),
                symptom_id=payload.get("symptom_id", fallback.symptom_id),
                symptom_text=payload.get(
                    "symptom_text", fallback.symptom_text
                ),
                date_start=payload.get("date_start", fallback.date_start),
                date_end=payload.get("date_end", fallback.date_end),
                max_safe_interval_days=int(
                    payload.get(
                        "max_safe_interval_days",
                        fallback.max_safe_interval_days,
                    )
                ),
                disease_id=payload.get("disease_id", fallback.disease_id),
                actions=actions or fallback.actions,
                planner="llm_constrained",
            )
        except Exception:
            return fallback

    def plan(self, question: str) -> QueryPlan:
        fallback = self.deterministic_plan(question)
        return self._llm_plan(question, fallback)
