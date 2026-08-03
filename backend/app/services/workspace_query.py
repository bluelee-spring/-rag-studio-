from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from ..models import EvidenceItem, QueryRequest, QueryResponse, TraceStage
from .answer_generator import AnswerGenerator
from .document_rag import DocumentRagService, FALLBACK_EMBEDDING_PROVIDER
from .model_runtime import model_runtime
from .orchestrator import MODE_NAMES
from .workspaces import WorkspaceRegistry
from .graph_workspace import WorkspaceGraphRagService


FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|attach|detach|pragma|vacuum|"
    r"create|replace|reindex|analyze|load_extension|transaction|commit|"
    r"rollback)\b",
    re.IGNORECASE,
)

ALLOWED_SQL_FUNCTIONS = {
    "abs",
    "avg",
    "coalesce",
    "count",
    "date",
    "datetime",
    "ifnull",
    "length",
    "lower",
    "max",
    "min",
    "nullif",
    "replace",
    "round",
    "strftime",
    "substr",
    "sum",
    "total",
    "trim",
    "upper",
}


def _extract_json(value: str) -> dict[str, Any]:
    cleaned = value.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    begin = cleaned.find("{")
    end = cleaned.rfind("}")
    if begin < 0 or end <= begin:
        raise ValueError("规划器没有返回JSON对象")
    payload = json.loads(cleaned[begin : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("规划器结果必须是JSON对象")
    return payload


def _validate_readonly_sql(sql: str) -> str:
    value = sql.strip().rstrip(";").strip()
    if not value:
        raise ValueError("SQL为空")
    without_strings = re.sub(r"'([^']|'')*'", "''", value)
    if ";" in without_strings:
        raise ValueError("只允许执行一个SQL语句")
    if not re.match(r"^(select|with)\b", value, re.IGNORECASE):
        raise ValueError("只允许SELECT或WITH查询")
    if FORBIDDEN_SQL.search(without_strings):
        raise ValueError("SQL包含非只读关键字")
    if not re.search(r"\brecords\b", value, re.IGNORECASE):
        raise ValueError("查询只能访问当前工作区的records表")
    return value


def _readonly_authorizer(
    action: int,
    argument1: str | None,
    argument2: str | None,
    _database: str | None,
    _trigger: str | None,
) -> int:
    """SQLite VM级边界：即使规划SQL绕过文本检查，也只能读records。"""
    if action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE}:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        return (
            sqlite3.SQLITE_OK
            if argument1 == "records"
            else sqlite3.SQLITE_DENY
        )
    if action == sqlite3.SQLITE_FUNCTION:
        function_name = (argument2 or argument1 or "").lower()
        return (
            sqlite3.SQLITE_OK
            if function_name in ALLOWED_SQL_FUNCTIONS
            else sqlite3.SQLITE_DENY
        )
    return sqlite3.SQLITE_DENY


class WorkspaceSqlRagService:
    def __init__(self, context: dict[str, Any]) -> None:
        self.manifest = context["manifest"]
        self.schema = context["schema"]
        self.database_path: Path = context["database_path"]
        self.columns = self.schema["columns"]
        self.by_sql = {item["sql_name"]: item for item in self.columns}

    def _schema_prompt(self) -> str:
        rows = [
            {
                "sql_name": item["sql_name"],
                "business_name": item["source_name"],
                "type": item["type"],
                "samples": item.get("samples", [])[:5],
            }
            for item in self.columns
        ]
        return json.dumps(
            {
                "table": "records",
                "row_count": self.schema["row_count"],
                "columns": rows,
            },
            ensure_ascii=False,
        )

    def _llm_plan(self, question: str) -> dict[str, Any]:
        output, provider, latency = model_runtime.generate(
            system=(
                "你是只读SQLite查询规划器。只能访问records表。"
                "输出严格JSON，不输出Markdown。"
            ),
            prompt=(
                "根据问题和字段映射生成只读参数化SQL。"
                "只能使用SELECT/WITH；值必须写成:p0、:p1等命名参数，"
                "并在parameters对象中给出值；禁止修改数据库。"
                "JSON字段：sql、parameters、intent、explanation、filters。\n"
                f"SCHEMA={self._schema_prompt()}\nQUESTION={question}"
            ),
            temperature=0,
            max_tokens=700,
        )
        payload = _extract_json(output)
        sql = _validate_readonly_sql(str(payload.get("sql", "")))
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("parameters必须是JSON对象")
        if not all(
            isinstance(value, (str, int, float, bool)) or value is None
            for value in parameters.values()
        ):
            raise ValueError("SQL参数只允许字符串、数字、布尔值或null")
        return {
            "planner": f"llm:{provider}",
            "planner_latency_ms": latency,
            "sql": sql,
            "parameters": parameters,
            "intent": str(payload.get("intent", "select")),
            "explanation": str(payload.get("explanation", "")),
            "filters": payload.get("filters", []),
        }

    def _mentioned_column(self, question: str) -> dict[str, Any] | None:
        lowered = question.lower()
        matches = [
            item
            for item in self.columns
            if str(item["source_name"]).lower() in lowered
            or item["sql_name"].lower() in lowered
        ]
        return max(
            matches,
            key=lambda item: len(str(item["source_name"])),
            default=None,
        )

    def _deterministic_plan(self, question: str) -> dict[str, Any]:
        lowered = question.lower()
        filters: list[dict[str, Any]] = []
        params: dict[str, Any] = {}
        for item in self.columns:
            for sample in item.get("samples", []):
                value = str(sample).strip()
                if len(value) < 2 or value.lower() not in lowered:
                    continue
                parameter = f"p{len(params)}"
                params[parameter] = sample
                filters.append(
                    {
                        "column": item["sql_name"],
                        "business_name": item["source_name"],
                        "operator": "=",
                        "parameter": parameter,
                        "value": sample,
                    }
                )
                break
        where = ""
        if filters:
            where = " WHERE " + " AND ".join(
                f'"{item["column"]}" = :{item["parameter"]}'
                for item in filters
            )

        column = self._mentioned_column(question)
        numeric_column = (
            column
            if column and column["type"] in {"INTEGER", "REAL"}
            else next(
                (
                    item
                    for item in self.columns
                    if item["type"] in {"INTEGER", "REAL"}
                    and str(item["source_name"]).lower() in lowered
                ),
                None,
            )
        )
        if any(token in lowered for token in ("多少", "几条", "数量", "总数", "count")):
            select = "COUNT(*) AS result_count"
            intent = "count"
        elif numeric_column and any(
            token in lowered for token in ("平均", "均值", "average", "avg")
        ):
            select = f'AVG("{numeric_column["sql_name"]}") AS result_average'
            intent = "average"
        elif numeric_column and any(
            token in lowered for token in ("总和", "合计", "sum")
        ):
            select = f'SUM("{numeric_column["sql_name"]}") AS result_sum'
            intent = "sum"
        elif numeric_column and any(
            token in lowered for token in ("最大", "最高", "max")
        ):
            select = f'MAX("{numeric_column["sql_name"]}") AS result_max'
            intent = "max"
        elif numeric_column and any(
            token in lowered for token in ("最小", "最低", "min")
        ):
            select = f'MIN("{numeric_column["sql_name"]}") AS result_min'
            intent = "min"
        else:
            selected = self.columns[:12]
            select = ", ".join(
                f'"{item["sql_name"]}"' for item in selected
            )
            intent = "select"
        return {
            "planner": "deterministic-schema-planner",
            "planner_latency_ms": 0,
            "sql": f'SELECT {select} FROM "records"{where}',
            "parameters": params,
            "intent": intent,
            "explanation": "根据字段名、样例值和聚合关键词构造只读查询。",
            "filters": filters,
        }

    def _plan(self, question: str) -> dict[str, Any]:
        if model_runtime.planner_enabled:
            try:
                return self._llm_plan(question)
            except Exception as exc:
                plan = self._deterministic_plan(question)
                plan["fallback_reason"] = str(exc)
                return plan
        return self._deterministic_plan(question)

    def run(
        self, question: str, top_k: int
    ) -> tuple[str, list[TraceStage], list[EvidenceItem], dict[str, Any]]:
        started = time.perf_counter()
        plan = self._plan(question)
        sql = _validate_readonly_sql(plan["sql"])
        parameters = dict(plan["parameters"])
        wrapped_sql = f"SELECT * FROM ({sql}) AS _rag_result LIMIT :_rag_limit"
        parameters["_rag_limit"] = max(20, top_k)

        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.set_authorizer(_readonly_authorizer)
        deadline = time.monotonic() + 3
        connection.set_progress_handler(
            lambda: 1 if time.monotonic() > deadline else 0,
            10_000,
        )
        try:
            explain = [
                dict(row)
                for row in connection.execute(
                    f"EXPLAIN QUERY PLAN {sql}",
                    plan["parameters"],
                ).fetchall()
            ]
            raw_rows = [
                dict(row)
                for row in connection.execute(
                    wrapped_sql, parameters
                ).fetchall()
            ]
        finally:
            connection.close()

        def business_key(key: str) -> str:
            return str(self.by_sql.get(key, {}).get("source_name", key))

        rows = [
            {business_key(key): value for key, value in row.items()}
            for row in raw_rows
        ]
        result_columns = list(rows[0]) if rows else []
        if plan["intent"] in {"count", "average", "sum", "max", "min"} and rows:
            value = next(iter(rows[0].values()))
            answer = (
                f"在“{self.manifest.name}”中，"
                f"按当前条件得到的{plan['intent']}结果为 {value}。"
            )
        else:
            answer = (
                f"在“{self.manifest.name}”中执行只读查询，"
                f"返回 {len(rows)} 行结果。"
            )
        evidence = [
            EvidenceItem(
                id=f"{self.manifest.id}-ROW-{index + 1:04d}",
                title=f"查询结果第{index + 1}行",
                excerpt=json.dumps(row, ensure_ascii=False, default=str),
                source=self.manifest.name,
            )
            for index, row in enumerate(rows[:top_k])
        ]
        elapsed = int((time.perf_counter() - started) * 1000)
        stages = [
            TraceStage(
                id="workspace-schema",
                title="读取工作区关系模式",
                kind="table-schema",
                data={
                    "workspace": self.manifest.model_dump(),
                    "table_name": "records",
                    "row_count": self.schema["row_count"],
                    "columns": self.columns,
                    "indexes": self.schema.get("indexes", []),
                    "preview": self.schema.get("preview", [])[:5],
                },
            ),
            TraceStage(
                id="workspace-sql-plan",
                title="自然语言约束映射为只读查询计划",
                kind="sql-plan",
                status=("warning" if plan.get("fallback_reason") else "completed"),
                description=plan["explanation"],
                data={key: value for key, value in plan.items() if key != "sql"},
            ),
            TraceStage(
                id="workspace-sql-code",
                title="查询计划编译为参数化SQL",
                kind="code",
                data={
                    "language": "sql",
                    "code": sql,
                    "parameters": plan["parameters"],
                    "safety": [
                        "仅SELECT/WITH",
                        "单语句",
                        "query_only连接",
                        "3秒执行预算",
                        "结果行数上限",
                    ],
                },
            ),
            TraceStage(
                id="workspace-sql-explain",
                title="SQLite生成实际执行计划",
                kind="query-plan",
                data={"rows": explain},
            ),
            TraceStage(
                id="workspace-sql-result",
                title="行集合进入证据上下文",
                kind="table-result",
                duration_ms=elapsed,
                data={
                    "columns": result_columns,
                    "rows": rows,
                    "row_count": len(rows),
                    "truncated": len(rows) >= max(20, top_k),
                },
            ),
        ]
        return answer, stages, evidence, {
            "latency_ms": elapsed,
            "planner": plan["planner"],
            "returned_rows": len(rows),
            "workspace_id": self.manifest.id,
        }


class WorkspaceRagRouter:
    def __init__(self, registry: WorkspaceRegistry) -> None:
        self.registry = registry
        self.answer_generator = AnswerGenerator()
        self._documents: dict[str, tuple[str, DocumentRagService]] = {}
        self._graphs: dict[
            str, tuple[str, str, WorkspaceGraphRagService]
        ] = {}
        self._lock = threading.Lock()

    def _document_service(self, workspace_id: str) -> DocumentRagService:
        manifest = self.registry.get(workspace_id)
        with self._lock:
            cached = self._documents.get(workspace_id)
            if cached and cached[0] == manifest.updated_at:
                return cached[1]
            service = DocumentRagService(
                self.registry.load_document_data(workspace_id)
            )
            self._documents[workspace_id] = (manifest.updated_at, service)
            return service

    def _graph_service(self, workspace_id: str) -> WorkspaceGraphRagService:
        manifest = self.registry.get(workspace_id)
        provider = (
            model_runtime.embedding_provider_name
            if model_runtime.embedding_ready
            else FALLBACK_EMBEDDING_PROVIDER
        )
        with self._lock:
            cached = self._graphs.get(workspace_id)
            if (
                cached
                and cached[0] == manifest.updated_at
                and cached[1] == provider
            ):
                return cached[2]
            service = WorkspaceGraphRagService(
                self.registry.graph_context(workspace_id)
            )
            self._graphs[workspace_id] = (
                manifest.updated_at,
                provider,
                service,
            )
            return service

    def run(self, request: QueryRequest) -> QueryResponse:
        manifest = self.registry.get(request.workspace_id)
        if request.mode not in manifest.supported_modes:
            raise ValueError(
                f"工作区“{manifest.name}”不支持{MODE_NAMES[request.mode]}"
            )
        if manifest.kind == "documents":
            service = self._document_service(request.workspace_id)
            if request.mode == "tfidf":
                answer, stages, evidence, metrics = service.tfidf(
                    request.question, request.top_k
                )
            elif request.mode == "bm25":
                answer, stages, evidence, metrics = service.bm25(
                    request.question, request.top_k
                )
            else:
                answer, stages, evidence, metrics = service.semantic(
                    request.question, request.top_k
                )
            source = f"文档工作区 · {manifest.name}"
        elif manifest.kind == "table":
            service = WorkspaceSqlRagService(
                self.registry.table_context(request.workspace_id)
            )
            answer, stages, evidence, metrics = service.run(
                request.question, request.top_k
            )
            source = f"SQLite工作区 · {manifest.name}"
        else:
            service = self._graph_service(request.workspace_id)
            answer, stages, evidence, metrics = service.run(
                request.question, request.top_k
            )
            source = f"内嵌属性图工作区 · {manifest.name}"

        answer, generation = self.answer_generator.generate(
            request.question, answer, evidence
        )
        stages.append(generation)
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
            disclaimer=(
                "回答仅依据当前用户工作区中的数据生成；"
                "请自行核验数据来源、权限与业务适用性。"
            ),
        )
