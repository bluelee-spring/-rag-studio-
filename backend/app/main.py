from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .data import teaching_data
from .models import (
    BatchClassifyRequest,
    BatchClassifyResponse,
    DatasetSummary,
    IngestionJobResponse,
    ModeInfo,
    ModelConfigRequest,
    ModelPublicStatus,
    ModelTestResponse,
    QueryRequest,
    QueryResponse,
    WorkspaceInfo,
)
from .services.model_runtime import model_runtime
from .services.orchestrator import RagOrchestrator
from .services.workspace_query import WorkspaceRagRouter
from .services.batch_classify import BatchClassifyService
from .services.workspaces import ingestion_manager, workspace_registry
from .services.workspaces import MAX_FILE_BYTES, MAX_UPLOAD_BYTES


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="面向课堂演示的多范式RAG执行与可视化接口。",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = RagOrchestrator(teaching_data)
workspace_router = WorkspaceRagRouter(workspace_registry)
batch_classify_service = BatchClassifyService(orchestrator, workspace_router)


async def _read_upload_limited(
    upload: UploadFile,
    *,
    total_so_far: int = 0,
) -> bytes:
    payload = bytearray()
    while chunk := await upload.read(1024 * 1024):
        payload.extend(chunk)
        if len(payload) > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"单个文件不能超过25MB：{upload.filename or 'upload'}",
            )
        if total_so_far + len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="单次上传总大小不能超过100MB",
            )
    return bytes(payload)


def _builtin_workspace() -> WorkspaceInfo:
    timestamp = datetime.fromtimestamp(
        teaching_data.sqlite_path.stat().st_mtime,
        tz=timezone.utc,
    ).isoformat()
    return WorkspaceInfo(
        id="builtin-soybean",
        name="内置大豆教学库",
        kind="builtin",
        status="ready",
        created_at=timestamp,
        updated_at=timestamp,
        source_files=["内置完整教学数据"],
        supported_modes=[
            "tfidf",
            "bm25",
            "semantic",
            "property_graph",
            "rdf",
            "sql",
            "composite",
        ],
        statistics={
            "documents": len(teaching_data.embedding_inputs),
            "chunks": len(teaching_data.chunks),
            "graph_nodes": len(teaching_data.graph_nodes),
            "graph_edges": len(teaching_data.graph_edges),
        },
        build={"managed": "system"},
    )


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": settings.app_version}


@app.get("/api/v1/modes", response_model=list[ModeInfo])
def modes() -> list[ModeInfo]:
    return [
        ModeInfo(
            id="tfidf",
            name="TF–IDF",
            family="文档检索",
            summary="词频、逆文档频率与余弦相似度",
            accent="#2F6B57",
        ),
        ModeInfo(
            id="bm25",
            name="BM25",
            family="文档检索",
            summary="词频饱和、长度归一化与稀疏排名",
            accent="#86642F",
        ),
        ModeInfo(
            id="semantic",
            name="语义向量",
            family="文档检索",
            summary="稠密表示与语义近邻",
            accent="#66558B",
        ),
        ModeInfo(
            id="property_graph",
            name="属性图",
            family="图检索",
            summary="实体锚点、Cypher与局部子图",
            accent="#315F76",
        ),
        ModeInfo(
            id="rdf",
            name="RDF / SPARQL",
            family="图检索",
            summary="IRI、本体词汇与证据三元组",
            accent="#8B4F4B",
        ),
        ModeInfo(
            id="sql",
            name="关系数据库",
            family="结构化查询",
            summary="查询计划、参数化SQL与精确聚合",
            accent="#4E5B66",
        ),
        ModeInfo(
            id="composite",
            name="综合路由",
            family="任务编排",
            summary="按任务选择检索器并合并证据",
            accent="#1F493D",
        ),
    ]


@app.get("/api/v1/dataset/summary", response_model=DatasetSummary)
def dataset_summary() -> DatasetSummary:
    with teaching_data.sqlite() as connection:
        cases = connection.execute(
            "SELECT COUNT(*) FROM field_case"
        ).fetchone()[0]
        documents = connection.execute(
            "SELECT COUNT(*) FROM document"
        ).fetchone()[0]
    return DatasetSummary(
        documents=documents,
        chunks=len(teaching_data.chunks),
        graph_nodes=len(teaching_data.graph_nodes),
        graph_edges=len(teaching_data.graph_edges),
        rdf_triples=len(orchestrator.rdf.graph),
        relational_cases=cases,
        external={
            "neo4j_configured": bool(settings.neo4j_password),
            "fuseki_configured": bool(settings.fuseki_query_url),
            "llm_configured": bool(
                model_runtime.generation_ready
            ),
            "embedding_api_configured": bool(
                model_runtime.embedding_ready
            ),
            "document_faiss_ready": (
                orchestrator.documents.vector_index.index_path.exists()
                and orchestrator.documents.vector_index.metadata_path.exists()
            ),
        },
    )


@app.get("/api/v1/workspaces", response_model=list[WorkspaceInfo])
def list_workspaces() -> list[WorkspaceInfo]:
    return [_builtin_workspace(), *workspace_registry.list()]


@app.get(
    "/api/v1/settings/model",
    response_model=ModelPublicStatus,
)
def model_status() -> ModelPublicStatus:
    return model_runtime.public_status()


@app.put(
    "/api/v1/settings/model",
    response_model=ModelPublicStatus,
)
def configure_model(request: ModelConfigRequest) -> ModelPublicStatus:
    return model_runtime.configure(request)


@app.post(
    "/api/v1/settings/model/test",
    response_model=ModelTestResponse,
)
def test_model_connection() -> ModelTestResponse:
    return model_runtime.test_connection()


@app.post(
    "/api/v1/workspaces/documents",
    response_model=IngestionJobResponse,
    status_code=202,
)
async def create_document_workspace(
    name: str = Form(..., min_length=1, max_length=100),
    chunk_size: int = Form(700, ge=300, le=2000),
    chunk_overlap: int = Form(100, ge=0, le=500),
    files: list[UploadFile] = File(...),
) -> IngestionJobResponse:
    if not 1 <= len(files) <= 10:
        raise HTTPException(status_code=400, detail="一次请上传1至10个文档")
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=400,
            detail="重叠长度必须小于目标分块长度",
        )
    payloads: list[tuple[str, bytes]] = []
    total = 0
    for file in files:
        content = await _read_upload_limited(file, total_so_far=total)
        total += len(content)
        payloads.append((file.filename or "document", content))
    try:
        return ingestion_manager.submit_documents(
            name=name,
            files=payloads,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/api/v1/workspaces/table",
    response_model=IngestionJobResponse,
    status_code=202,
)
async def create_table_workspace(
    name: str = Form(..., min_length=1, max_length=100),
    sheet_name: str = Form("", max_length=100),
    file: UploadFile = File(...),
) -> IngestionJobResponse:
    try:
        content = await _read_upload_limited(file)
        return ingestion_manager.submit_table(
            name=name,
            file=(file.filename or "table", content),
            sheet_name=sheet_name.strip(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/api/v1/workspaces/graph",
    response_model=IngestionJobResponse,
    status_code=202,
)
async def create_graph_workspace(
    name: str = Form(..., min_length=1, max_length=100),
    file: UploadFile = File(...),
) -> IngestionJobResponse:
    try:
        content = await _read_upload_limited(file)
        return ingestion_manager.submit_graph(
            name=name,
            file=(file.filename or "graph.json", content),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get(
    "/api/v1/ingestion/jobs/{job_id}",
    response_model=IngestionJobResponse,
)
def ingestion_job(job_id: str) -> IngestionJobResponse:
    try:
        return ingestion_manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/document-index/status")
def document_index_status() -> dict:
    metadata = orchestrator.documents.vector_index.ensure()
    return {
        "status": "ready",
        "index_type": metadata["index_type"],
        "search_mode": metadata["search_mode"],
        "provider": metadata["provider"],
        "dimensions": metadata["dimensions"],
        "vector_count": metadata["vector_count"],
        "load_source": metadata["load_source"],
        "index_file": (
            orchestrator.documents.vector_index.index_path.name
        ),
        "metadata_file": (
            orchestrator.documents.vector_index.metadata_path.name
        ),
    }


@app.post("/api/v1/rag/query", response_model=QueryResponse)
def run_query(request: QueryRequest) -> QueryResponse:
    try:
        if request.workspace_id != "builtin-soybean":
            return workspace_router.run(request)
        return orchestrator.run(request)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except sqlite3.DatabaseError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"关系数据库执行失败：{exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"检索执行失败：{exc}",
        ) from exc


@app.post("/api/v1/batch/classify", response_model=BatchClassifyResponse)
def run_batch_classify(request: BatchClassifyRequest) -> BatchClassifyResponse:
    try:
        return batch_classify_service.run(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"批量分类执行失败：{exc}",
        ) from exc
