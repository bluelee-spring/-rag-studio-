from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np

from ..data import TeachingData


class DenseEncoderProtocol(Protocol):
    @property
    def provider_name(self) -> str: ...

    def encode(self, text: str) -> list[float]: ...

    def encode_many(self, texts: list[str]) -> list[list[float]]: ...

    def set_expected_dimension(self, dimension: int) -> None: ...


class DocumentFaissIndex:
    """文档向量的持久化FAISS索引与行号元数据。"""

    def __init__(
        self,
        data: TeachingData,
        encoder: DenseEncoderProtocol,
    ):
        self.data = data
        self.encoder = encoder
        self.index_path = (
            data.sqlite_path.parents[1]
            / "documents"
            / "document_embeddings.faiss"
        )
        self.metadata_path = (
            data.sqlite_path.parents[1]
            / "documents"
            / "document_embeddings.meta.json"
        )
        self._index: faiss.Index | None = None
        self._metadata: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def _corpus_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for item in self.data.embedding_inputs:
            digest.update(item["id"].encode("utf-8"))
            digest.update(b"\0")
            digest.update(item["text"].encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _load(self) -> bool:
        if not self.index_path.exists() or not self.metadata_path.exists():
            return False
        try:
            metadata = json.loads(
                self.metadata_path.read_text(encoding="utf-8")
            )
            if metadata["corpus_fingerprint"] != self._corpus_fingerprint():
                return False
            if metadata["provider"] != self.encoder.provider_name:
                return False
            if metadata["chunk_ids"] != [
                item["id"] for item in self.data.embedding_inputs
            ]:
                return False

            # Windows版FAISS的C++文件接口无法稳定处理中文路径。
            # 由Python负责文件读写，再从内存反序列化，可兼容中文用户名。
            serialized = np.frombuffer(
                self.index_path.read_bytes(),
                dtype=np.uint8,
            )
            index = faiss.deserialize_index(serialized)
            if index.ntotal != len(self.data.embedding_inputs):
                return False
            if index.d != int(metadata["dimensions"]):
                return False

            self.encoder.set_expected_dimension(index.d)
            self._index = index
            self._metadata = {**metadata, "load_source": "disk"}
            return True
        except (OSError, ValueError, KeyError, RuntimeError):
            return False

    def _persist(
        self,
        index: faiss.Index,
        metadata: dict[str, Any],
    ) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_index = self.index_path.with_suffix(".faiss.tmp")
        temporary_metadata = self.metadata_path.with_suffix(".json.tmp")
        serialized = faiss.serialize_index(index)
        temporary_index.write_bytes(serialized.tobytes())
        temporary_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_index.replace(self.index_path)
        temporary_metadata.replace(self.metadata_path)

    def _build(self) -> dict[str, Any]:
        texts = [
            item["text"] for item in self.data.embedding_inputs
        ]
        vectors = self.encoder.encode_many(texts)
        if not vectors:
            raise RuntimeError("文档语料为空，无法构建FAISS索引")

        dimensions = len(vectors[0])
        if any(len(vector) != dimensions for vector in vectors):
            raise RuntimeError("Embedding返回了不一致的向量维度")

        matrix = np.asarray(vectors, dtype=np.float32)
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(dimensions)
        index.add(matrix)

        metadata: dict[str, Any] = {
            "format_version": 1,
            "index_type": "IndexFlatIP",
            "search_mode": "exact",
            "similarity": "inner_product_after_l2_normalization",
            "provider": self.encoder.provider_name,
            "dimensions": dimensions,
            "vector_count": int(index.ntotal),
            "chunk_ids": [
                item["id"] for item in self.data.embedding_inputs
            ],
            "corpus_fingerprint": self._corpus_fingerprint(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._persist(index, metadata)
        self.encoder.set_expected_dimension(dimensions)
        self._index = index
        self._metadata = {**metadata, "load_source": "rebuilt"}
        return self._metadata

    def ensure(self, force: bool = False) -> dict[str, Any]:
        with self._lock:
            if not force and self._index is not None and self._metadata:
                if self._metadata.get("provider") == self.encoder.provider_name:
                    return self._metadata
                self._index = None
                self._metadata = None
            if not force and self._load():
                return self._metadata or {}
            return self._build()

    def search(
        self,
        question: str,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
        metadata = self.ensure()
        if self._index is None:
            raise RuntimeError("FAISS索引尚未加载")

        query_vector = self.encoder.encode(question)
        if len(query_vector) != self._index.d:
            raise RuntimeError(
                "查询向量维度与文档索引不一致；请重新构建FAISS索引"
            )
        query_matrix = np.asarray([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_matrix)
        scores, positions = self._index.search(
            query_matrix,
            min(top_k, int(self._index.ntotal)),
        )

        results: list[dict[str, Any]] = []
        chunk_ids = metadata["chunk_ids"]
        for score, position in zip(
            scores[0],
            positions[0],
            strict=True,
        ):
            if position < 0:
                continue
            results.append(
                {
                    "chunk_id": chunk_ids[int(position)],
                    "position": int(position),
                    "score": float(score),
                    "vector": self._index.reconstruct(
                        int(position)
                    ).astype(float).tolist(),
                }
            )
        return results, query_matrix[0].astype(float).tolist(), metadata
