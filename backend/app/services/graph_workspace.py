from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np

from ..data import TeachingData
from ..models import EvidenceItem, TraceStage, WorkspaceInfo
from .document_rag import (
    DenseEncoder,
    FALLBACK_EMBEDDING_PROVIDER,
    QueryTokenizer,
    _group_vector_products,
    _project_vector_3d,
)


GRAPH_EXTENSIONS = {".json", ".jsonl"}
MAX_GRAPH_NODES = 50_000
MAX_GRAPH_EDGES = 200_000
MAX_GRAPH_DOCUMENTS = 20_000
MAX_GRAPH_PROPERTY_BYTES = 64 * 1024
MAX_GRAPH_TEXT_CHARACTERS = 20_000
MAX_GRAPH_HOPS = 2
MAX_EXPANDED_NODES = 100
MAX_EXPANDED_EDGES = 180
MAX_GRAPH_NODE_TYPES = 256
MAX_GRAPH_EDGE_TYPES = 512


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("图文件必须使用UTF-8、UTF-8 BOM或GB18030编码")


def _record_type(row: dict[str, Any]) -> str:
    return str(row.get("record_type") or row.get("kind") or "").strip().lower()


def _parse_source(path: Path) -> dict[str, Any]:
    text = _read_text(path)
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"graph.json解析失败：第{exc.lineno}行第{exc.colno}列，{exc.msg}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError("graph.json顶层必须是JSON对象")
        return payload

    payload: dict[str, Any] = {
        "graph_version": "1.0",
        "schema": {},
        "nodes": [],
        "edges": [],
        "documents": [],
    }
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"graph.jsonl第{line_number}行解析失败：{exc.msg}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"graph.jsonl第{line_number}行必须是JSON对象")
        kind = _record_type(row)
        clean = {
            key: value
            for key, value in row.items()
            if key not in {"record_type", "kind"}
        }
        if kind == "schema":
            payload["schema"] = clean
        elif kind == "node":
            payload["nodes"].append(clean)
        elif kind == "edge":
            payload["edges"].append(clean)
        elif kind == "document":
            payload["documents"].append(clean)
        elif kind == "graph":
            payload.update(clean)
        else:
            raise ValueError(
                f"graph.jsonl第{line_number}行record_type必须是"
                "schema、node、edge、document或graph"
            )
    return payload


def _identifier(value: Any, label: str, *, maximum: int = 200) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{label}不能为空")
    if len(result) > maximum:
        raise ValueError(f"{label}不能超过{maximum}个字符")
    if re.search(r"[\x00-\x1f]", result):
        raise ValueError(f"{label}不能包含控制字符")
    return result


def _properties(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label}.properties必须是JSON对象")
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.properties包含不可序列化值") from exc
    if len(encoded.encode("utf-8")) > MAX_GRAPH_PROPERTY_BYTES:
        raise ValueError(
            f"{label}.properties不能超过{MAX_GRAPH_PROPERTY_BYTES // 1024}KB"
        )
    return value


def _evidence_ids(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label}.evidence_ids必须是字符串数组")
    return list(dict.fromkeys(_identifier(item, label) for item in value))


def _schema_definition(
    raw_schema: Any,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    schema = raw_schema if isinstance(raw_schema, dict) else {}
    if raw_schema is not None and not isinstance(raw_schema, dict):
        raise ValueError("schema必须是JSON对象")
    if schema.get("node_types") is not None and not isinstance(
        schema["node_types"], list
    ):
        raise ValueError("schema.node_types必须是数组")
    if schema.get("edge_types") is not None and not isinstance(
        schema["edge_types"], list
    ):
        raise ValueError("schema.edge_types必须是数组")
    if len(schema.get("node_types") or []) > MAX_GRAPH_NODE_TYPES:
        raise ValueError(f"节点类型不能超过{MAX_GRAPH_NODE_TYPES}种")
    if len(schema.get("edge_types") or []) > MAX_GRAPH_EDGE_TYPES:
        raise ValueError(f"关系类型不能超过{MAX_GRAPH_EDGE_TYPES}种")
    node_schema_supplied = bool(schema.get("node_types"))
    edge_schema_supplied = bool(schema.get("edge_types"))
    supplied = node_schema_supplied or edge_schema_supplied
    node_types: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(schema.get("node_types") or []):
        if not isinstance(item, dict):
            raise ValueError(f"schema.node_types[{index}]必须是对象")
        name = _identifier(item.get("name"), "节点类型name", maximum=100)
        if name in node_types:
            raise ValueError(f"节点类型重复：{name}")
        node_types[name] = {
            "name": name,
            "label": str(item.get("label") or name).strip()[:100],
        }
    for node_type in sorted({node["type"] for node in nodes}):
        if node_schema_supplied and node_type not in node_types:
            raise ValueError(f"节点使用了schema未声明的类型：{node_type}")
        node_types.setdefault(node_type, {"name": node_type, "label": node_type})

    edge_types: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(schema.get("edge_types") or []):
        if not isinstance(item, dict):
            raise ValueError(f"schema.edge_types[{index}]必须是对象")
        name = _identifier(item.get("name"), "关系类型name", maximum=100)
        if name in edge_types:
            raise ValueError(f"关系类型重复：{name}")
        raw_source_types = item.get("source_types") or []
        raw_target_types = item.get("target_types") or []
        if not isinstance(raw_source_types, list) or not isinstance(
            raw_target_types, list
        ):
            raise ValueError(
                f"关系类型{name}的source_types和target_types必须是数组"
            )
        source_types = [
            _identifier(value, f"关系类型{name}.source_types", maximum=100)
            for value in raw_source_types
        ]
        target_types = [
            _identifier(value, f"关系类型{name}.target_types", maximum=100)
            for value in raw_target_types
        ]
        edge_types[name] = {
            "name": name,
            "label": str(item.get("label") or name).strip()[:100],
            "source_types": source_types,
            "target_types": target_types,
        }
    for edge_type in sorted({edge["type"] for edge in edges}):
        if edge_schema_supplied and edge_type not in edge_types:
            raise ValueError(f"边使用了schema未声明的类型：{edge_type}")
        edge_types.setdefault(
            edge_type,
            {
                "name": edge_type,
                "label": edge_type,
                "source_types": sorted(
                    {
                        edge["source_type"]
                        for edge in edges
                        if edge["type"] == edge_type
                    }
                ),
                "target_types": sorted(
                    {
                        edge["target_type"]
                        for edge in edges
                        if edge["type"] == edge_type
                    }
                ),
            },
        )
    if len(node_types) > MAX_GRAPH_NODE_TYPES:
        raise ValueError(f"推断出的节点类型不能超过{MAX_GRAPH_NODE_TYPES}种")
    if len(edge_types) > MAX_GRAPH_EDGE_TYPES:
        raise ValueError(f"推断出的关系类型不能超过{MAX_GRAPH_EDGE_TYPES}种")
    known_node_types = set(node_types)
    for definition in edge_types.values():
        unknown = (
            set(definition["source_types"])
            | set(definition["target_types"])
        ) - known_node_types
        if unknown:
            raise ValueError(
                f"关系类型{definition['name']}引用了未声明节点类型："
                + "、".join(sorted(unknown))
            )
    return {
        "format_version": 1,
        "node_types": list(node_types.values()),
        "edge_types": list(edge_types.values()),
    }, supplied


def _normalize_graph(payload: dict[str, Any]) -> dict[str, Any]:
    raw_nodes = payload.get("nodes")
    raw_edges = payload.get("edges")
    raw_documents = payload.get("documents") or []
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("图文件必须包含非空nodes数组")
    if not isinstance(raw_edges, list):
        raise ValueError("图文件必须包含edges数组")
    if not isinstance(raw_documents, list):
        raise ValueError("documents必须是数组")
    if len(raw_nodes) > MAX_GRAPH_NODES:
        raise ValueError(f"节点数不能超过{MAX_GRAPH_NODES:,}")
    if len(raw_edges) > MAX_GRAPH_EDGES:
        raise ValueError(f"边数不能超过{MAX_GRAPH_EDGES:,}")
    if len(raw_documents) > MAX_GRAPH_DOCUMENTS:
        raise ValueError(f"证据文档数不能超过{MAX_GRAPH_DOCUMENTS:,}")

    nodes: list[dict[str, Any]] = []
    node_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            raise ValueError(f"nodes[{index}]必须是对象")
        node_id = _identifier(raw.get("id"), f"nodes[{index}].id")
        if node_id in node_by_id:
            raise ValueError(f"节点ID重复：{node_id}")
        node_type = _identifier(
            raw.get("type"), f"nodes[{index}].type", maximum=100
        )
        name = _identifier(raw.get("name"), f"nodes[{index}].name", maximum=500)
        text = str(raw.get("text") or name).strip()
        if len(text) > MAX_GRAPH_TEXT_CHARACTERS:
            raise ValueError(f"节点{node_id}的text过长")
        node = {
            "id": node_id,
            "type": node_type,
            "name": name,
            "text": text,
            "properties": _properties(raw.get("properties"), f"节点{node_id}"),
            "evidence_ids": _evidence_ids(
                raw.get("evidence_ids"), f"节点{node_id}"
            ),
        }
        nodes.append(node)
        node_by_id[node_id] = node

    edges: list[dict[str, Any]] = []
    edge_ids: set[str] = set()
    for index, raw in enumerate(raw_edges):
        if not isinstance(raw, dict):
            raise ValueError(f"edges[{index}]必须是对象")
        edge_id = _identifier(
            raw.get("id") or f"edge_{index + 1:06d}",
            f"edges[{index}].id",
        )
        if edge_id in edge_ids:
            raise ValueError(f"边ID重复：{edge_id}")
        source = _identifier(raw.get("source"), f"边{edge_id}.source")
        target = _identifier(raw.get("target"), f"边{edge_id}.target")
        if source not in node_by_id:
            raise ValueError(f"边{edge_id}的source节点不存在：{source}")
        if target not in node_by_id:
            raise ValueError(f"边{edge_id}的target节点不存在：{target}")
        edge_type = _identifier(raw.get("type"), f"边{edge_id}.type", maximum=100)
        edge = {
            "id": edge_id,
            "source": source,
            "target": target,
            "type": edge_type,
            "text": str(
                raw.get("text")
                or f"{node_by_id[source]['name']} -[{edge_type}]-> {node_by_id[target]['name']}"
            ).strip()[:MAX_GRAPH_TEXT_CHARACTERS],
            "properties": _properties(raw.get("properties"), f"边{edge_id}"),
            "evidence_ids": _evidence_ids(
                raw.get("evidence_ids"), f"边{edge_id}"
            ),
            "source_type": node_by_id[source]["type"],
            "target_type": node_by_id[target]["type"],
        }
        edges.append(edge)
        edge_ids.add(edge_id)

    documents: list[dict[str, str]] = []
    document_ids: set[str] = set()
    for index, raw in enumerate(raw_documents):
        if not isinstance(raw, dict):
            raise ValueError(f"documents[{index}]必须是对象")
        document_id = _identifier(raw.get("id"), f"documents[{index}].id")
        if document_id in document_ids:
            raise ValueError(f"证据文档ID重复：{document_id}")
        title = _identifier(
            raw.get("title"), f"documents[{index}].title", maximum=500
        )
        text = str(raw.get("text") or "").strip()
        if not text:
            raise ValueError(f"证据文档{document_id}.text不能为空")
        if len(text) > MAX_GRAPH_TEXT_CHARACTERS:
            raise ValueError(f"证据文档{document_id}.text过长")
        documents.append(
            {
                "id": document_id,
                "title": title,
                "text": text,
                "source": str(raw.get("source") or title).strip()[:1000],
            }
        )
        document_ids.add(document_id)

    schema, schema_supplied = _schema_definition(
        payload.get("schema"), nodes, edges
    )
    edge_definitions = {item["name"]: item for item in schema["edge_types"]}
    for edge in edges:
        definition = edge_definitions[edge["type"]]
        if definition["source_types"] and edge["source_type"] not in definition["source_types"]:
            raise ValueError(
                f"边{edge['id']}的起点类型{edge['source_type']}不符合schema"
            )
        if definition["target_types"] and edge["target_type"] not in definition["target_types"]:
            raise ValueError(
                f"边{edge['id']}的终点类型{edge['target_type']}不符合schema"
            )

    warnings: list[str] = []
    referenced = {
        evidence_id
        for item in [*nodes, *edges]
        for evidence_id in item["evidence_ids"]
    }
    missing_evidence = sorted(referenced - document_ids)
    if missing_evidence:
        warnings.append(
            "以下evidence_id没有对应documents记录："
            + "、".join(missing_evidence[:20])
        )
    degree: Counter[str] = Counter()
    for edge in edges:
        degree[edge["source"]] += 1
        degree[edge["target"]] += 1
    isolated = [node["id"] for node in nodes if degree[node["id"]] == 0]
    if isolated:
        warnings.append(f"检测到{len(isolated)}个孤立节点")

    return {
        "graph_version": str(payload.get("graph_version") or "1.0"),
        "name": str(payload.get("name") or "").strip(),
        "description": str(payload.get("description") or "").strip(),
        "nodes": nodes,
        "edges": edges,
        "documents": documents,
        "schema": schema,
        "schema_supplied": schema_supplied,
        "warnings": warnings,
        "isolated_node_ids": isolated,
    }


def _graph_tokens(text: str) -> list[str]:
    normalized = text.lower()
    try:
        import jieba

        values = jieba.lcut(normalized, cut_all=False)
    except ImportError:
        values = re.findall(r"[a-z][a-z0-9_+-]*|[\u4e00-\u9fff]{1,4}", normalized)
    tokens = [value.strip() for value in values if value.strip()]
    return tokens or [normalized[:32]]


def _node_cards(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> list[dict[str, str]]:
    node_by_id = {node["id"]: node for node in nodes}
    relations: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        source = node_by_id[edge["source"]]
        target = node_by_id[edge["target"]]
        relations[source["id"]].append(
            f"出边：{edge['type']} → {target['name']}"
        )
        relations[target["id"]].append(
            f"入边：{source['name']} → {edge['type']}"
        )
    cards: list[dict[str, str]] = []
    for node in nodes:
        properties = "；".join(
            f"{key}={value}" for key, value in node["properties"].items()
        )
        parts = [
            f"节点类型：{node['type']}",
            f"名称：{node['name']}",
            f"描述：{node['text']}",
        ]
        if properties:
            parts.append("属性：" + properties)
        parts.extend(relations[node["id"]][:12])
        cards.append({"id": node["id"], "text": "\n".join(parts)})
    return cards


def _embedding_data(
    root: Path,
    cards: list[dict[str, str]],
) -> TeachingData:
    tokenized = [
        {"chunk_id": item["id"], "tokens": _graph_tokens(item["text"])}
        for item in cards
    ]
    return TeachingData(
        chunks=[],
        embedding_inputs=cards,
        tokenized_chunks=tokenized,
        synonyms={},
        graph_nodes={},
        graph_edges=[],
        sqlite_path=root / "graph" / "graph.sqlite3",
        ontology_path=root / "graph" / "ontology.ttl",
        instances_path=root / "graph" / "instances.ttl",
        shapes_path=root / "graph" / "shapes.ttl",
    )


def _build_faiss(root: Path, cards: list[dict[str, str]]) -> dict[str, Any]:
    data = _embedding_data(root, cards)
    encoder = DenseEncoder(QueryTokenizer(data))
    vectors = encoder.encode_many([item["text"] for item in cards])
    if not vectors:
        raise ValueError("没有可建立向量索引的节点")
    dimensions = len(vectors[0])
    if any(len(vector) != dimensions for vector in vectors):
        raise ValueError("节点Embedding维度不一致")
    matrix = np.asarray(vectors, dtype=np.float32)
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(dimensions)
    index.add(matrix)
    graph_dir = root / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    serialized = faiss.serialize_index(index)
    (graph_dir / "node_embeddings.faiss").write_bytes(serialized.tobytes())
    fingerprint = hashlib.sha256(
        "\0".join(f"{item['id']}\0{item['text']}" for item in cards).encode("utf-8")
    ).hexdigest()
    metadata = {
        "format_version": 1,
        "index_type": "IndexFlatIP",
        "search_mode": "exact",
        "similarity": "inner_product_after_l2_normalization",
        "provider": encoder.provider_name,
        "dimensions": dimensions,
        "vector_count": int(index.ntotal),
        "node_ids": [item["id"] for item in cards],
        "corpus_fingerprint": fingerprint,
        "created_at": _now(),
    }
    _write_json(graph_dir / "node_embeddings.meta.json", metadata)
    return metadata


def build_graph_workspace(
    *,
    registry_root: Path,
    workspace_id: str,
    name: str,
    source_path: Path,
    update: Callable[[str, int, dict[str, Any] | None], None],
) -> WorkspaceInfo:
    building = registry_root / f".{workspace_id}.building"
    final = registry_root / workspace_id
    building.mkdir(parents=True, exist_ok=False)
    created = _now()
    try:
        update("parse-graph", 14, {"source": source_path.name})
        source_dir = building / "source"
        source_dir.mkdir(parents=True)
        target = source_dir / Path(source_path.name).name
        shutil.copy2(source_path, target)
        payload = _parse_source(target)

        update("validate-graph", 30, None)
        graph = _normalize_graph(payload)
        nodes = graph["nodes"]
        edges = graph["edges"]
        documents = graph["documents"]
        update(
            "validate-graph",
            36,
            {
                "valid": True,
                "nodes": len(nodes),
                "edges": len(edges),
                "documents": len(documents),
                "warnings": graph["warnings"],
            },
        )

        update("sqlite-graph", 52, {"nodes": len(nodes), "edges": len(edges)})
        graph_dir = building / "graph"
        graph_dir.mkdir(parents=True)
        database_path = graph_dir / "graph.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE graph_node (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    text TEXT NOT NULL,
                    properties_json TEXT NOT NULL
                );
                CREATE TABLE graph_edge (
                    edge_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES graph_node(node_id),
                    target_id TEXT NOT NULL REFERENCES graph_node(node_id),
                    edge_type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    properties_json TEXT NOT NULL
                );
                CREATE TABLE graph_document (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL
                );
                CREATE TABLE node_evidence (
                    node_id TEXT NOT NULL REFERENCES graph_node(node_id),
                    document_id TEXT NOT NULL REFERENCES graph_document(document_id),
                    PRIMARY KEY (node_id, document_id)
                );
                CREATE TABLE edge_evidence (
                    edge_id TEXT NOT NULL REFERENCES graph_edge(edge_id),
                    document_id TEXT NOT NULL REFERENCES graph_document(document_id),
                    PRIMARY KEY (edge_id, document_id)
                );
                CREATE INDEX idx_graph_node_type ON graph_node(node_type);
                CREATE INDEX idx_graph_node_name ON graph_node(name);
                CREATE INDEX idx_graph_edge_source ON graph_edge(source_id, edge_type);
                CREATE INDEX idx_graph_edge_target ON graph_edge(target_id, edge_type);
                CREATE INDEX idx_graph_edge_type ON graph_edge(edge_type);
                """
            )
            connection.executemany(
                "INSERT INTO graph_node VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        node["id"], node["type"], node["name"], node["text"],
                        json.dumps(node["properties"], ensure_ascii=False),
                    )
                    for node in nodes
                ],
            )
            connection.executemany(
                "INSERT INTO graph_edge VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        edge["id"], edge["source"], edge["target"], edge["type"],
                        edge["text"], json.dumps(edge["properties"], ensure_ascii=False),
                    )
                    for edge in edges
                ],
            )
            connection.executemany(
                "INSERT INTO graph_document VALUES (?, ?, ?, ?)",
                [(item["id"], item["title"], item["text"], item["source"]) for item in documents],
            )
            known_documents = {item["id"] for item in documents}
            connection.executemany(
                "INSERT INTO node_evidence VALUES (?, ?)",
                [
                    (node["id"], evidence_id)
                    for node in nodes
                    for evidence_id in node["evidence_ids"]
                    if evidence_id in known_documents
                ],
            )
            connection.executemany(
                "INSERT INTO edge_evidence VALUES (?, ?)",
                [
                    (edge["id"], evidence_id)
                    for edge in edges
                    for evidence_id in edge["evidence_ids"]
                    if evidence_id in known_documents
                ],
            )
            connection.commit()
        finally:
            connection.close()

        update("node-cards", 68, None)
        cards = _node_cards(nodes, edges)
        update("node-cards", 72, {"cards": len(cards)})
        _write_jsonl(graph_dir / "nodes.jsonl", nodes)
        _write_jsonl(graph_dir / "edges.jsonl", edges)
        _write_jsonl(graph_dir / "documents.jsonl", documents)
        _write_jsonl(graph_dir / "node_cards.jsonl", cards)
        _write_json(
            graph_dir / "schema.json",
            {
                **graph["schema"],
                "graph_version": graph["graph_version"],
                "name": graph["name"],
                "description": graph["description"],
                "schema_supplied": graph["schema_supplied"],
            },
        )
        _write_json(
            graph_dir / "validation_report.json",
            {
                "valid": True,
                "warnings": graph["warnings"],
                "node_count": len(nodes),
                "edge_count": len(edges),
                "document_count": len(documents),
                "isolated_node_ids": graph["isolated_node_ids"][:500],
            },
        )

        update("node-vectors", 82, {"cards": len(cards)})
        vector_metadata = _build_faiss(building, cards)
        update(
            "node-vectors",
            88,
            {
                "provider": vector_metadata["provider"],
                "dimensions": vector_metadata["dimensions"],
                "vectors": vector_metadata["vector_count"],
            },
        )
        update("finalize-graph", 94, {"provider": vector_metadata["provider"]})
        manifest = WorkspaceInfo(
            id=workspace_id,
            name=name.strip() or graph["name"] or "未命名图工作区",
            kind="graph",
            status="ready",
            created_at=created,
            updated_at=_now(),
            source_files=[target.name],
            supported_modes=["property_graph"],
            statistics={
                "nodes": len(nodes),
                "edges": len(edges),
                "documents": len(documents),
                "node_types": len(graph["schema"]["node_types"]),
                "edge_types": len(graph["schema"]["edge_types"]),
                "isolated_nodes": len(graph["isolated_node_ids"]),
                "vectors": vector_metadata["vector_count"],
                "dimensions": vector_metadata["dimensions"],
                "warnings": len(graph["warnings"]),
            },
            build={
                "storage": "SQLite property graph",
                "anchor_index": "FAISS IndexFlatIP",
                "embedding_provider": vector_metadata["provider"],
                "max_hops": MAX_GRAPH_HOPS,
                "schema_mode": "declared" if graph["schema_supplied"] else "inferred",
                "artifacts": [
                    "graph.sqlite3", "schema.json", "validation_report.json",
                    "node_cards.jsonl", "node_embeddings.faiss",
                    "node_embeddings.meta.json",
                ],
            },
        )
        _write_json(building / "workspace.json", manifest.model_dump())
        building.replace(final)
        update("completed", 100, manifest.statistics)
        return manifest
    except Exception:
        if building.exists():
            shutil.rmtree(building)
        raise


def load_graph_context(root: Path, manifest: WorkspaceInfo) -> dict[str, Any]:
    graph_dir = root / "graph"
    return {
        "manifest": manifest,
        "database_path": graph_dir / "graph.sqlite3",
        "schema": json.loads((graph_dir / "schema.json").read_text(encoding="utf-8")),
        "cards": [
            json.loads(line)
            for line in (graph_dir / "node_cards.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
        "index_path": graph_dir / "node_embeddings.faiss",
        "metadata_path": graph_dir / "node_embeddings.meta.json",
    }


class WorkspaceGraphRagService:
    def __init__(self, context: dict[str, Any]) -> None:
        self.manifest: WorkspaceInfo = context["manifest"]
        self.database_path: Path = context["database_path"]
        self.schema: dict[str, Any] = context["schema"]
        self.cards: list[dict[str, str]] = context["cards"]
        self.card_by_id = {item["id"]: item for item in self.cards}
        data = _embedding_data(self.database_path.parents[1], self.cards)
        self.encoder = DenseEncoder(QueryTokenizer(data))
        metadata = json.loads(context["metadata_path"].read_text(encoding="utf-8"))
        if metadata["provider"] != self.encoder.provider_name:
            raise ValueError(
                "当前Embedding配置与该图工作区建库时不同；请用当前模型重新上传构建工作区"
            )
        serialized = np.frombuffer(context["index_path"].read_bytes(), dtype=np.uint8)
        self.index = faiss.deserialize_index(serialized)
        self.metadata = metadata
        self.encoder.set_expected_dimension(self.index.d)
        (
            self.nodes,
            self.documents,
            self.node_evidence,
            self.edge_evidence,
        ) = self._load_rows()
        self.position_by_node_id = {
            node_id: index
            for index, node_id in enumerate(self.metadata["node_ids"])
        }

    def _load_rows(self) -> tuple[
        dict[str, dict[str, Any]], dict[str, dict[str, Any]],
        dict[str, set[str]], dict[str, set[str]],
    ]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            nodes = {
                row["node_id"]: {
                    "id": row["node_id"], "type": row["node_type"],
                    "name": row["name"], "text": row["text"],
                    "properties": json.loads(row["properties_json"]),
                }
                for row in connection.execute("SELECT * FROM graph_node")
            }
            documents = {
                row["document_id"]: dict(row)
                for row in connection.execute("SELECT * FROM graph_document")
            }
            node_evidence: dict[str, set[str]] = defaultdict(set)
            for row in connection.execute("SELECT * FROM node_evidence"):
                node_evidence[row["node_id"]].add(row["document_id"])
            edge_evidence: dict[str, set[str]] = defaultdict(set)
            for row in connection.execute("SELECT * FROM edge_evidence"):
                edge_evidence[row["edge_id"]].add(row["document_id"])
            return nodes, documents, node_evidence, edge_evidence
        finally:
            connection.close()

    def _adjacent_edges(
        self,
        frontier: set[str],
        allowed: set[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        if not frontier or not allowed or limit <= 0:
            return []
        node_ids = sorted(frontier)
        relationships = sorted(allowed)
        relation_slots = ",".join("?" for _ in relationships)
        node_slots = ",".join("?" for _ in node_ids)
        sql = (
            "SELECT edge_id, source_id, target_id, edge_type, text, "
            "properties_json FROM graph_edge "
            f"WHERE edge_type IN ({relation_slots}) AND "
            f"(source_id IN ({node_slots}) OR target_id IN ({node_slots})) "
            "ORDER BY edge_type, edge_id LIMIT ?"
        )
        parameters = [*relationships, *node_ids, *node_ids, limit]
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        try:
            return [
                {
                    "id": row["edge_id"],
                    "source": row["source_id"],
                    "target": row["target_id"],
                    "type": row["edge_type"],
                    "text": row["text"],
                    "properties": json.loads(row["properties_json"]),
                }
                for row in connection.execute(sql, parameters)
            ]
        finally:
            connection.close()

    def _anchors(self, question: str, top_k: int) -> tuple[list[dict[str, Any]], list[float]]:
        query_vector = self.encoder.encode(question)
        query = np.asarray([query_vector], dtype=np.float32)
        faiss.normalize_L2(query)
        search_k = min(max(top_k * 3, 12), int(self.index.ntotal))
        scores, positions = self.index.search(query, search_k)
        candidates: list[dict[str, Any]] = []
        for score, position in zip(scores[0], positions[0], strict=True):
            if position < 0:
                continue
            node_id = self.metadata["node_ids"][int(position)]
            node = self.nodes[node_id]
            vector = self.index.reconstruct(int(position)).astype(float).tolist()
            lexical_boost = 0.0
            if node["name"] in question:
                lexical_boost = 0.35
            elif len(question) >= 2 and question in node["name"]:
                lexical_boost = 0.2
            candidates.append(
                {
                    "id": node_id, "name": node["name"], "type": node["type"],
                    "description": node["text"], "vector_score": float(score),
                    "lexical_boost": lexical_boost,
                    "score": float(score) + lexical_boost,
                    "projection": _project_vector_3d(vector),
                    "product_groups": _group_vector_products(
                        query[0].astype(float).tolist(), vector, group_count=16
                    ),
                }
            )
        retrieved_ids = {item["id"] for item in candidates}
        exact_matches = sorted(
            (
                node
                for node in self.nodes.values()
                if len(node["name"]) >= 2 and node["name"] in question
                and node["id"] not in retrieved_ids
            ),
            key=lambda node: (-len(node["name"]), node["name"]),
        )[:20]
        for node in exact_matches:
            position = self.position_by_node_id[node["id"]]
            vector = self.index.reconstruct(position).astype(float).tolist()
            vector_score = float(np.dot(query[0], np.asarray(vector, dtype=np.float32)))
            candidates.append(
                {
                    "id": node["id"],
                    "name": node["name"],
                    "type": node["type"],
                    "description": node["text"],
                    "vector_score": vector_score,
                    "lexical_boost": 0.35,
                    "score": vector_score + 0.35,
                    "projection": _project_vector_3d(vector),
                    "product_groups": _group_vector_products(
                        query[0].astype(float).tolist(), vector, group_count=16
                    ),
                }
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[: max(top_k, 5)], query[0].astype(float).tolist()

    def _allowed_relations(self, question: str) -> list[str]:
        matched: list[str] = []
        lowered = question.casefold()
        for item in self.schema["edge_types"]:
            terms = [str(item["name"]), str(item.get("label") or "")]
            if any(term and term.casefold() in lowered for term in terms):
                matched.append(str(item["name"]))
        if matched:
            return matched
        cues = {
            "症状": ("symptom", "症状", "表现"),
            "药剂": ("chemical", "pesticide", "treat", "药剂", "防治", "推荐"),
            "地点": ("location", "region", "地区", "地点", "发生"),
            "病害": ("disease", "diagnos", "病害", "诊断"),
        }
        active_cues = [key for key in cues if key in question]
        for item in self.schema["edge_types"]:
            haystack = " ".join(
                [str(item["name"]), str(item.get("label") or ""),
                 *map(str, item.get("source_types") or []),
                 *map(str, item.get("target_types") or [])]
            ).casefold()
            if any(
                token in haystack
                for cue in active_cues
                for token in cues[cue]
            ):
                matched.append(str(item["name"]))
        return matched or [str(item["name"]) for item in self.schema["edge_types"]]

    def run(
        self, question: str, top_k: int
    ) -> tuple[str, list[TraceStage], list[EvidenceItem], dict[str, Any]]:
        started = time.perf_counter()
        candidates, query_vector = self._anchors(question, top_k)
        if not candidates:
            raise ValueError("图工作区没有可召回的实体锚点")
        anchors = candidates[: min(3, len(candidates))]
        anchor_ids = [item["id"] for item in anchors]
        allowed = set(self._allowed_relations(question))
        selected_nodes = set(anchor_ids)
        selected_edges: dict[str, dict[str, Any]] = {}
        frontier = set(anchor_ids)
        steps: list[dict[str, Any]] = []
        for depth in range(1, MAX_GRAPH_HOPS + 1):
            next_frontier: set[str] = set()
            input_count = len(frontier)
            adjacent = self._adjacent_edges(
                frontier,
                allowed,
                MAX_EXPANDED_EDGES - len(selected_edges),
            )
            for edge in adjacent:
                if edge["source"] in frontier:
                    neighbor = edge["target"]
                else:
                    neighbor = edge["source"]
                selected_edges[edge["id"]] = edge
                if neighbor not in selected_nodes:
                    next_frontier.add(neighbor)
                selected_nodes.add(neighbor)
                if (
                    len(selected_nodes) >= MAX_EXPANDED_NODES
                    or len(selected_edges) >= MAX_EXPANDED_EDGES
                ):
                    break
            steps.append(
                {
                    "index": depth,
                    "operation": f"沿关系白名单执行第{depth}跳局部扩展",
                    "relation": " | ".join(sorted(allowed)[:6]),
                    "input_count": input_count,
                    "output_count": len(next_frontier),
                    "active_ids": sorted(next_frontier)[:80],
                }
            )
            frontier = next_frontier
            if not frontier:
                break

        evidence_ids: list[str] = []
        for node_id in sorted(selected_nodes):
            evidence_ids.extend(sorted(self.node_evidence.get(node_id, set())))
        for edge_id in selected_edges:
            evidence_ids.extend(sorted(self.edge_evidence.get(edge_id, set())))
        evidence_ids = list(dict.fromkeys(evidence_ids))
        evidence: list[EvidenceItem] = []
        for document_id in evidence_ids[:top_k]:
            document = self.documents.get(document_id)
            if document:
                evidence.append(
                    EvidenceItem(
                        id=document_id,
                        title=document["title"],
                        excerpt=document["text"],
                        source=document["source"],
                    )
                )
        if not evidence:
            for candidate in candidates[:top_k]:
                node = self.nodes[candidate["id"]]
                evidence.append(
                    EvidenceItem(
                        id=node["id"], title=f"{node['type']} · {node['name']}",
                        excerpt=node["text"], source=f"{self.manifest.name} · 图节点",
                        score=round(float(candidate["score"]), 4),
                    )
                )

        graph_nodes = [
            {"id": node_id, "label": self.nodes[node_id]["name"], "type": self.nodes[node_id]["type"]}
            for node_id in sorted(selected_nodes)
        ]
        graph_edges = [
            {"source": edge["source"], "target": edge["target"], "relation": edge["type"]}
            for edge in selected_edges.values()
        ]
        type_definitions = {
            item["name"]: item for item in self.schema["node_types"]
        }
        allowed_definitions = [
            item for item in self.schema["edge_types"] if item["name"] in allowed
        ]
        related_types = list(
            dict.fromkeys(
                node_type
                for item in allowed_definitions
                for node_type in [
                    *item.get("source_types", []),
                    *item.get("target_types", []),
                ]
                if node_type not in {self.nodes[item_id]["type"] for item_id in anchor_ids}
            )
        )[:3]
        type_nodes = [
            {
                "id": f"type-{node_type}",
                "label": type_definitions.get(node_type, {}).get("label") or node_type,
                "role": "target",
            }
            for node_type in related_types
        ]
        elapsed = int((time.perf_counter() - started) * 1000)
        stages = [
            TraceStage(
                id="workspace-entity-anchor",
                title="查询向量在节点语义空间中寻找锚点",
                kind="entity-space",
                data={
                    "question": question,
                    "query_label": "用户问题",
                    "query_projection": _project_vector_3d(query_vector),
                    "candidates": candidates,
                    "selected": anchors[0]["id"],
                    "dimensions": self.metadata["dimensions"],
                    "provider": self.metadata["provider"],
                },
            ),
            TraceStage(
                id="workspace-graph-boundary",
                title="问题意图编译为受约束图模式",
                kind="graph-pattern",
                data={
                    "pattern_nodes": [
                        {"id": "anchor", "label": anchors[0]["name"], "role": "anchor"},
                        *type_nodes,
                    ],
                    "pattern_edges": [
                        {"relation": relation} for relation in sorted(allowed)[:3]
                    ],
                    "filters": [
                        {"field": "anchor", "operator": "IN", "value": "semantic top-3"},
                        {"field": "edge.type", "operator": "IN", "value": "relationship whitelist"},
                        {"field": "path.length", "operator": "<=", "value": MAX_GRAPH_HOPS},
                    ],
                    "allowed_relationships": sorted(allowed),
                    "max_hops": MAX_GRAPH_HOPS,
                    "workspace_schema": self.schema,
                },
            ),
            TraceStage(
                id="workspace-graph-traversal",
                title="SQLite邻接索引执行逐跳局部扩展",
                kind="graph-traversal",
                duration_ms=elapsed,
                data={
                    "anchor_id": anchors[0]["id"],
                    "anchor_ids": anchor_ids,
                    "steps": steps,
                    "graph": {"nodes": graph_nodes, "edges": graph_edges},
                    "storage": "SQLite adjacency indexes",
                    "truncated": len(selected_nodes) >= MAX_EXPANDED_NODES,
                },
            ),
            TraceStage(
                id="workspace-graph-evidence",
                title="图路径关联证据进入上下文",
                kind="ranking",
                data={
                    "results": [
                        {
                            "id": item.id, "title": item.title,
                            "excerpt": item.excerpt,
                            "score": item.score if item.score is not None else 1.0,
                        }
                        for item in evidence
                    ],
                    "selected_nodes": len(selected_nodes),
                    "selected_edges": len(selected_edges),
                },
            ),
        ]
        relation_facts = [
            f"{self.nodes[edge['source']]['name']} -[{edge['type']}]-> "
            f"{self.nodes[edge['target']]['name']}"
            for edge in list(selected_edges.values())[:8]
        ]
        answer = (
            f"在“{self.manifest.name}”中，语义锚点为"
            + "、".join(item["name"] for item in anchors)
            + "。局部子图事实："
            + ("；".join(relation_facts) if relation_facts else "没有命中允许关系")
            + f"。共回收{len(evidence)}条证据；最终回答仅应依据这些证据生成。"
        )
        return answer, stages, evidence, {
            "latency_ms": elapsed,
            "workspace_id": self.manifest.id,
            "anchor_ids": anchor_ids,
            "allowed_relationships": sorted(allowed),
            "expanded_nodes": len(selected_nodes),
            "expanded_edges": len(selected_edges),
            "embedding_provider": self.metadata["provider"],
            "dimensions": self.metadata["dimensions"],
        }
