import json
import sqlite3
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.models import ModelConfigRequest, QueryRequest
from app.services.model_runtime import RuntimeModelManager
from app.services.vector_index import DocumentFaissIndex
from app.services.graph_workspace import _parse_source
from app.services.workspace_query import (
    WorkspaceRagRouter,
    _readonly_authorizer,
)
from app.services.workspaces import IngestionManager, WorkspaceRegistry


client = TestClient(app)

QUESTION = (
    "2025年7月六合区的大豆病例中，叶片出现褐色近圆形病斑时，"
    "最可能是什么病？共有多少例？还常伴随哪些症状？"
    "推荐一种安全间隔期不超过14天的药剂，并给出依据。"
)


def test_health() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_dataset_summary() -> None:
    response = client.get("/api/v1/dataset/summary")
    assert response.status_code == 200
    payload = response.json()
    assert payload["chunks"] == 228
    assert payload["graph_nodes"] == 372
    assert payload["graph_edges"] == 1567
    assert payload["relational_cases"] == 240


def test_document_modes() -> None:
    for mode in ("tfidf", "bm25", "semantic"):
        response = client.post(
            "/api/v1/rag/query",
            json={"mode": mode, "question": QUESTION, "top_k": 5},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["stages"]
        assert payload["evidence"]
        generation = next(
            stage
            for stage in payload["stages"]
            if stage["kind"] == "generation"
        )
        assert generation["data"]["prompt_layers"]
        assert generation["data"]["evidence_cards"]
        assert generation["data"]["final_answer"] == payload["answer"]
        assert isinstance(generation["data"]["llm_enabled"], bool)

        kinds = [stage["kind"] for stage in payload["stages"]]
        if mode == "tfidf":
            assert kinds[:4] == [
                "tokens",
                "vocabulary-space",
                "tfidf-build",
                "cosine-workbench",
            ]
            build = payload["stages"][2]["data"]
            comparison = payload["stages"][3]["data"]["comparisons"][0]
            assert build["dimensions"] == 110
            assert len(build["raw_vector"]) == 110
            assert len(build["normalized_vector"]) == 110
            assert len(comparison["query_vector"]) == 110
            assert len(comparison["document_vector"]) == 110
            assert "numerator" in comparison
            assert "score" in comparison
        elif mode == "bm25":
            assert kinds[:4] == [
                "tokens",
                "bm25-corpus",
                "bm25-document",
                "bm25-accumulator",
            ]
            document = payload["stages"][2]["data"]["documents"][0]
            term = document["terms"][0]
            assert {
                "idf",
                "length_normalizer",
                "denominator",
                "saturation",
                "contribution",
            } <= term.keys()
        else:
            assert kinds[:5] == [
                "text",
                "vector-index",
                "dense-vector",
                "vector-space",
                "dense-similarity",
            ]
            comparison = payload["stages"][4]["data"]["comparisons"][0]
            assert len(comparison["query_vector"]) == 192
            assert len(comparison["document_vector"]) == 192
            assert comparison["groups"]


def test_semantic_mode_uses_persisted_faiss() -> None:
    status = client.get("/api/v1/document-index/status")
    assert status.status_code == 200
    assert status.json()["index_type"] == "IndexFlatIP"
    assert status.json()["vector_count"] == 228

    response = client.post(
        "/api/v1/rag/query",
        json={"mode": "semantic", "question": QUESTION, "top_k": 5},
    )
    payload = response.json()
    index_stage = next(
        stage
        for stage in payload["stages"]
        if stage["kind"] == "vector-index"
    )
    assert index_stage["data"]["vector_count"] == 228
    assert payload["metrics"]["index_type"] == "IndexFlatIP"


def test_structured_modes_return_gold_count() -> None:
    for mode in ("sql", "property_graph", "rdf", "composite"):
        response = client.post(
            "/api/v1/rag/query",
            json={"mode": mode, "question": QUESTION, "top_k": 5},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert "24" in payload["answer"], (mode, payload["answer"])

        kinds = [stage["kind"] for stage in payload["stages"]]
        if mode == "property_graph":
            assert kinds[:4] == [
                "entity-space",
                "graph-pattern",
                "graph-traversal",
                "graph-aggregate",
            ]
            traversal = payload["stages"][2]["data"]
            assert [
                step["output_count"] for step in traversal["steps"]
            ] == [74, 34, 24, 3, 3]
            assert traversal["graph"]["nodes"]
            assert traversal["graph"]["edges"]
        elif mode == "rdf":
            assert kinds[:4] == [
                "ontology-space",
                "iri-mapping",
                "triple-pattern",
                "rdf-filter",
            ]
            candidates = payload["stages"][3]["data"]["candidates"]
            assert any(item["passed"] for item in candidates)
            assert any(not item["passed"] for item in candidates)
        elif mode == "sql":
            assert kinds[:4] == [
                "relational-plan",
                "row-filter",
                "key-join",
                "group-aggregate",
            ]
            filters = payload["stages"][1]["data"]["filters"]
            assert filters[-1]["count"] == 24
            assert payload["stages"][2]["data"]["sample_rows"]


class _TinyEncoder:
    provider_name = "unicode-path-test"

    def __init__(self) -> None:
        self.dimension: int | None = None

    def encode(self, text: str) -> list[float]:
        return [1.0, 0.5, 0.25]

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, float(index + 1), 0.25]
            for index, _ in enumerate(texts)
        ]

    def set_expected_dimension(self, dimension: int) -> None:
        self.dimension = dimension


def test_faiss_persistence_supports_unicode_path(tmp_path) -> None:
    data_root = tmp_path / "卢航青教学数据"
    sqlite_path = data_root / "relational" / "soybean.sqlite3"
    data = SimpleNamespace(
        sqlite_path=sqlite_path,
        embedding_inputs=[
            {"id": "DOC-1", "text": "褐色近圆形病斑"},
            {"id": "DOC-2", "text": "大豆褐斑病"},
        ],
    )
    encoder = _TinyEncoder()
    first = DocumentFaissIndex(data, encoder)
    metadata = first.ensure(force=True)
    assert metadata["vector_count"] == 2
    assert first.index_path.exists()

    second = DocumentFaissIndex(data, encoder)
    loaded = second.ensure()
    assert loaded["load_source"] == "disk"
    results, query, _ = second.search("近圆形病斑", 2)
    assert len(results) == 2
    assert len(query) == 3
    assert len(results[0]["vector"]) == 3


def test_runtime_model_status_never_returns_plaintext_key() -> None:
    manager = RuntimeModelManager()
    status = manager.configure(
        ModelConfigRequest(
            provider="remote_api",
            api_base="https://gateway.example/v1/chat/completions",
            api_key="classroom-secret-placeholder",
            chat_model="classroom-model",
            embedding_model="embedding-model",
            enable_answer=True,
        )
    )
    payload = status.model_dump_json()
    assert status.api_key_present is True
    assert status.api_base == "https://gateway.example/v1/chat/completions"
    assert "classroom-secret-placeholder" not in payload
    assert "secret-placeholder" not in payload


def _write_fake_full_model(path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"teaching-placeholder")


def test_local_full_or_merged_model_is_detected_from_one_path(tmp_path) -> None:
    model_path = tmp_path / "models" / "Qwen2.5-0.5B-Instruct"
    _write_fake_full_model(model_path)
    manager = RuntimeModelManager()
    status = manager.configure(
        ModelConfigRequest(
            provider="local_huggingface",
            local_model_path=f'"{model_path}"',
            enable_answer=True,
        )
    )
    assert status.configured is True
    assert status.local_artifact_type == "full_model"
    assert status.resolved_model_path == str(model_path)
    assert status.resolved_adapter_path == ""


def test_lora_run_path_auto_resolves_sibling_base_model(tmp_path) -> None:
    backend = tmp_path / "lora-visual-lab" / "backend"
    model_path = backend / "models" / "Qwen2.5-0.5B-Instruct"
    run_path = backend / "outputs" / "run_8b0aca5164f7"
    _write_fake_full_model(model_path)
    run_path.mkdir(parents=True)
    (run_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "peft_type": "LORA",
            }
        ),
        encoding="utf-8",
    )
    (run_path / "adapter_model.safetensors").write_bytes(
        b"teaching-placeholder"
    )

    manager = RuntimeModelManager()
    status = manager.configure(
        ModelConfigRequest(
            provider="local_huggingface",
            local_model_path=str(run_path),
            enable_answer=True,
        )
    )
    assert status.configured is True
    assert status.local_artifact_type == "lora_adapter"
    assert status.resolved_model_path == str(model_path)
    assert status.resolved_adapter_path == str(run_path)
    assert any("自动定位基础模型" in note for note in status.notes)


def test_document_workspace_builds_three_retrieval_assets(tmp_path) -> None:
    registry = WorkspaceRegistry(tmp_path / "workspaces")
    source = tmp_path / "小麦病害手册.txt"
    source.write_text(
        "小麦条锈病会在叶片形成黄色条状孢子堆。"
        "田间调查应记录发生区域、发病日期和叶片症状。\n\n"
        "赤霉病常在穗部出现粉红色霉层，应结合天气和抽穗期判断。",
        encoding="utf-8",
    )
    workspace = registry.build_documents(
        workspace_id="ws-doc-a1b2c3d4e5",
        name="小麦病害手册",
        source_paths=[source],
        chunk_size=300,
        chunk_overlap=60,
        update=lambda _stage, _progress, _details: None,
    )
    documents = registry.root / workspace.id / "documents"
    assert workspace.statistics["documents"] == 1
    assert workspace.statistics["chunks"] >= 1
    assert (documents / "lexical_index.json").exists()
    assert (documents / "document_embeddings.faiss").exists()
    assert (documents / "document_embeddings.meta.json").exists()

    response = WorkspaceRagRouter(registry).run(
        QueryRequest(
            workspace_id=workspace.id,
            mode="semantic",
            question="叶片黄色条状孢子堆是什么病",
            top_k=3,
        )
    )
    assert response.evidence
    assert response.execution_source == "文档工作区 · 小麦病害手册"
    assert any(stage.kind == "vector-index" for stage in response.stages)


def test_csv_workspace_becomes_readonly_sql_rag(tmp_path) -> None:
    registry = WorkspaceRegistry(tmp_path / "workspaces")
    source = tmp_path / "田间记录.csv"
    source.write_text(
        "地区,作物,产量\n六合区,大豆,10\n六合区,大豆,12\n浦口区,玉米,8\n",
        encoding="utf-8-sig",
    )
    workspace = registry.build_table(
        workspace_id="ws-tab-f1e2d3c4b5",
        name="田间记录",
        source_path=source,
        sheet_name="",
        update=lambda _stage, _progress, _details: None,
    )
    context = registry.table_context(workspace.id)
    assert context["schema"]["row_count"] == 3
    assert [column["source_name"] for column in context["schema"]["columns"]] == [
        "地区",
        "作物",
        "产量",
    ]

    response = WorkspaceRagRouter(registry).run(
        QueryRequest(
            workspace_id=workspace.id,
            mode="sql",
            question="六合区共有多少条记录",
            top_k=5,
        )
    )
    assert "2" in response.answer
    assert [stage.kind for stage in response.stages[:5]] == [
        "table-schema",
        "sql-plan",
        "code",
        "query-plan",
        "table-result",
    ]
    assert response.stages[2].data["code"].lower().startswith("select")


def test_sqlite_authorizer_blocks_schema_exfiltration() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute('CREATE TABLE "records" ("c_001" TEXT)')
    connection.execute('INSERT INTO "records" VALUES ("allowed")')
    connection.set_authorizer(_readonly_authorizer)
    assert connection.execute(
        'SELECT "c_001" FROM "records"'
    ).fetchone()[0] == "allowed"
    with pytest.raises(sqlite3.DatabaseError):
        connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'records'"
        ).fetchall()
    connection.close()


def test_multipart_upload_job_publishes_queryable_workspace(
    tmp_path, monkeypatch
) -> None:
    registry = WorkspaceRegistry(tmp_path / "workspaces")
    manager = IngestionManager(registry)
    manager.staging = tmp_path / "ingestion"
    manager.staging.mkdir(parents=True)
    monkeypatch.setattr(main_module, "workspace_registry", registry)
    monkeypatch.setattr(main_module, "ingestion_manager", manager)
    monkeypatch.setattr(
        main_module, "workspace_router", WorkspaceRagRouter(registry)
    )
    try:
        response = client.post(
            "/api/v1/workspaces/table",
            data={"name": "接口上传测试", "sheet_name": ""},
            files={
                "file": (
                    "巡检.csv",
                    "地区,状态\n六合区,正常\n六合区,异常\n".encode(
                        "utf-8-sig"
                    ),
                    "text/csv",
                )
            },
        )
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]
        payload = response.json()
        for _ in range(100):
            payload = client.get(
                f"/api/v1/ingestion/jobs/{job_id}"
            ).json()
            if payload["status"] in {"completed", "failed"}:
                break
            time.sleep(0.02)
        assert payload["status"] == "completed", payload
        assert not (manager.staging / job_id).exists()

        query = client.post(
            "/api/v1/rag/query",
            json={
                "workspace_id": payload["workspace_id"],
                "mode": "sql",
                "question": "六合区共有多少条记录",
                "top_k": 3,
            },
        )
        assert query.status_code == 200, query.text
        assert "2" in query.json()["answer"]
    finally:
        manager.executor.shutdown(wait=True)


def test_excel_sheet_is_inferred_and_imported(tmp_path) -> None:
    from openpyxl import Workbook

    source = tmp_path / "教学产量.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "产量表"
    sheet.append(["地区", "产量", "备注"])
    sheet.append(["六合区", 12.5, "复核"])
    sheet.append(["浦口区", 9.0, None])
    workbook.save(source)
    workbook.close()

    registry = WorkspaceRegistry(tmp_path / "workspaces")
    workspace = registry.build_table(
        workspace_id="ws-tab-0123456789",
        name="Excel产量",
        source_path=source,
        sheet_name="产量表",
        update=lambda _stage, _progress, _details: None,
    )
    schema = registry.table_context(workspace.id)["schema"]
    assert schema["source_detail"] == "产量表"
    assert schema["row_count"] == 2
    assert [column["type"] for column in schema["columns"]] == [
        "TEXT",
        "REAL",
        "TEXT",
    ]


def _graph_payload() -> dict:
    return {
        "graph_version": "1.0",
        "name": "大豆病害知识图",
        "schema": {
            "node_types": [
                {"name": "Disease", "label": "病害"},
                {"name": "Symptom", "label": "症状"},
                {"name": "Chemical", "label": "药剂"},
            ],
            "edge_types": [
                {
                    "name": "HAS_SYMPTOM",
                    "label": "具有症状",
                    "source_types": ["Disease"],
                    "target_types": ["Symptom"],
                },
                {
                    "name": "TREATED_BY",
                    "label": "推荐药剂",
                    "source_types": ["Disease"],
                    "target_types": ["Chemical"],
                },
            ],
        },
        "nodes": [
            {
                "id": "disease_001",
                "type": "Disease",
                "name": "大豆褐斑病",
                "text": "主要危害叶片。",
                "properties": {"crop": "大豆"},
                "evidence_ids": ["doc_001"],
            },
            {
                "id": "symptom_001",
                "type": "Symptom",
                "name": "褐色近圆形病斑",
                "text": "叶片出现褐色近圆形病斑。",
                "evidence_ids": ["doc_001"],
            },
            {
                "id": "chemical_001",
                "type": "Chemical",
                "name": "苯醚甲环唑",
                "text": "安全间隔期14天。",
                "properties": {"safe_interval_days": 14},
                "evidence_ids": ["doc_002"],
            },
        ],
        "edges": [
            {
                "id": "edge_001",
                "source": "disease_001",
                "target": "symptom_001",
                "type": "HAS_SYMPTOM",
                "evidence_ids": ["doc_001"],
            },
            {
                "id": "edge_002",
                "source": "disease_001",
                "target": "chemical_001",
                "type": "TREATED_BY",
                "evidence_ids": ["doc_002"],
            },
        ],
        "documents": [
            {
                "id": "doc_001",
                "title": "症状依据",
                "text": "大豆褐斑病可形成褐色近圆形病斑。",
                "source": "教学资料",
            },
            {
                "id": "doc_002",
                "title": "药剂依据",
                "text": "苯醚甲环唑的教学安全间隔期为14天。",
                "source": "教学资料",
            },
        ],
    }


def test_graph_workspace_builds_embedded_graph_and_faiss(tmp_path) -> None:
    registry = WorkspaceRegistry(tmp_path / "workspaces")
    source = tmp_path / "graph.json"
    source.write_text(
        json.dumps(_graph_payload(), ensure_ascii=False), encoding="utf-8"
    )
    workspace = registry.build_graph(
        workspace_id="ws-gra-a1b2c3d4e5",
        name="大豆图工作区",
        source_path=source,
        update=lambda _stage, _progress, _details: None,
    )
    graph_dir = registry.root / workspace.id / "graph"
    assert workspace.kind == "graph"
    assert workspace.supported_modes == ["property_graph"]
    assert workspace.statistics["nodes"] == 3
    assert workspace.statistics["edges"] == 2
    assert workspace.statistics["vectors"] == 3
    for filename in (
        "graph.sqlite3",
        "schema.json",
        "validation_report.json",
        "node_cards.jsonl",
        "node_embeddings.faiss",
        "node_embeddings.meta.json",
    ):
        assert (graph_dir / filename).exists(), filename

    connection = sqlite3.connect(graph_dir / "graph.sqlite3")
    try:
        assert connection.execute("SELECT COUNT(*) FROM graph_node").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM graph_edge").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM node_evidence").fetchone()[0] == 3
    finally:
        connection.close()

    response = WorkspaceRagRouter(registry).run(
        QueryRequest(
            workspace_id=workspace.id,
            mode="property_graph",
            question="褐色近圆形病斑最可能是什么病，推荐什么药剂？",
            top_k=3,
        )
    )
    assert response.evidence
    assert response.execution_source == "内嵌属性图工作区 · 大豆图工作区"
    assert [stage.kind for stage in response.stages[:4]] == [
        "entity-space",
        "graph-pattern",
        "graph-traversal",
        "ranking",
    ]
    traversal = response.stages[2].data
    assert traversal["graph"]["nodes"]
    assert traversal["graph"]["edges"]
    assert len(traversal["steps"]) <= 2
    assert response.stages[-1].kind == "generation"


def test_graph_jsonl_contract_is_parsed(tmp_path) -> None:
    source = tmp_path / "graph.jsonl"
    payload = _graph_payload()
    rows = [
        {"record_type": "schema", **payload["schema"]},
        *(
            {"record_type": "node", **item}
            for item in payload["nodes"]
        ),
        *(
            {"record_type": "edge", **item}
            for item in payload["edges"]
        ),
        *(
            {"record_type": "document", **item}
            for item in payload["documents"]
        ),
    ]
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    parsed = _parse_source(source)
    assert len(parsed["nodes"]) == 3
    assert len(parsed["edges"]) == 2
    assert parsed["schema"]["edge_types"][0]["name"] == "HAS_SYMPTOM"


def test_graph_validation_rejects_dangling_edge_atomically(tmp_path) -> None:
    registry = WorkspaceRegistry(tmp_path / "workspaces")
    payload = _graph_payload()
    payload["edges"][0]["target"] = "missing_node"
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="target节点不存在"):
        registry.build_graph(
            workspace_id="ws-gra-0123456789",
            name="无效图",
            source_path=source,
            update=lambda _stage, _progress, _details: None,
        )
    assert not (registry.root / "ws-gra-0123456789").exists()
    assert not (registry.root / ".ws-gra-0123456789.building").exists()


def test_graph_multipart_job_publishes_queryable_workspace(
    tmp_path, monkeypatch
) -> None:
    registry = WorkspaceRegistry(tmp_path / "workspaces")
    manager = IngestionManager(registry)
    manager.staging = tmp_path / "ingestion"
    manager.staging.mkdir(parents=True)
    monkeypatch.setattr(main_module, "workspace_registry", registry)
    monkeypatch.setattr(main_module, "ingestion_manager", manager)
    monkeypatch.setattr(
        main_module, "workspace_router", WorkspaceRagRouter(registry)
    )
    try:
        response = client.post(
            "/api/v1/workspaces/graph",
            data={"name": "接口图上传测试"},
            files={
                "file": (
                    "graph.json",
                    json.dumps(_graph_payload(), ensure_ascii=False).encode("utf-8"),
                    "application/json",
                )
            },
        )
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]
        payload = response.json()
        for _ in range(150):
            payload = client.get(f"/api/v1/ingestion/jobs/{job_id}").json()
            if payload["status"] in {"completed", "failed"}:
                break
            time.sleep(0.02)
        assert payload["status"] == "completed", payload
        assert len(payload["stages"]) == 8
        assert not (manager.staging / job_id).exists()

        query = client.post(
            "/api/v1/rag/query",
            json={
                "workspace_id": payload["workspace_id"],
                "mode": "property_graph",
                "question": "褐色近圆形病斑是什么病？",
                "top_k": 3,
            },
        )
        assert query.status_code == 200, query.text
        assert query.json()["evidence"]
        assert query.json()["metrics"]["expanded_nodes"] >= 2
    finally:
        manager.executor.shutdown(wait=True)
