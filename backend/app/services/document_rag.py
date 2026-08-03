from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from typing import Any

from ..data import TeachingData
from ..models import EvidenceItem, TraceStage
from .model_runtime import model_runtime
from .vector_index import DocumentFaissIndex


FALLBACK_EMBEDDING_PROVIDER = "deterministic-teaching-encoder-v3"


def _l2_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _normalize(values: list[float]) -> list[float]:
    norm = _l2_norm(values)
    if norm == 0:
        return values
    return [value / norm for value in values]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _project_vector_3d(values: list[float]) -> dict[str, float]:
    """把高维向量稳定投影到三个展示轴；检索分数仍由完整向量计算。"""
    axes = [0.0, 0.0, 0.0]
    for index, value in enumerate(values):
        axis = index % 3
        direction = 1.0 if (index // 3) % 2 == 0 else -1.0
        axes[axis] += value * direction
    scale = max(max(abs(value) for value in axes), 1e-9)
    return {
        "x": round(axes[0] / scale, 5),
        "y": round(axes[1] / scale, 5),
        "z": round(axes[2] / scale, 5),
    }


def _group_vector_products(
    query: list[float],
    document: list[float],
    group_count: int = 24,
) -> list[dict[str, float | int]]:
    """把逐维乘积聚合成可视分组，同时保留完整点积值。"""
    group_size = max(1, math.ceil(len(query) / group_count))
    groups: list[dict[str, float | int]] = []
    for begin in range(0, len(query), group_size):
        end = min(len(query), begin + group_size)
        query_slice = query[begin:end]
        document_slice = document[begin:end]
        groups.append(
            {
                "from": begin,
                "to": end - 1,
                "query_energy": round(
                    sum(value * value for value in query_slice),
                    6,
                ),
                "document_energy": round(
                    sum(value * value for value in document_slice),
                    6,
                ),
                "product": round(
                    sum(
                        left * right
                        for left, right in zip(
                            query_slice,
                            document_slice,
                            strict=True,
                        )
                    ),
                    6,
                ),
            }
        )
    return groups


class QueryTokenizer:
    def __init__(self, data: TeachingData):
        self.data = data
        self.vocabulary = data.token_vocabulary
        self.alias_to_canonical: dict[str, str] = {}
        for canonical, aliases in data.synonyms.items():
            self.alias_to_canonical[canonical] = canonical
            for alias in aliases:
                self.alias_to_canonical[alias] = canonical

    def tokenize(self, text: str) -> tuple[list[str], list[dict[str, str]]]:
        matches: list[tuple[int, int, str]] = []
        for term in self.vocabulary:
            start = 0
            while True:
                position = text.find(term, start)
                if position < 0:
                    break
                matches.append((position, position + len(term), term))
                start = position + len(term)

        matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        occupied: set[int] = set()
        accepted: list[tuple[int, int, str]] = []
        for begin, end, term in matches:
            span = set(range(begin, end))
            if span & occupied:
                continue
            occupied.update(span)
            accepted.append((begin, end, term))

        accepted.sort()
        tokens: list[str] = []
        normalizations: list[dict[str, str]] = []
        for _, _, term in accepted:
            canonical = self.alias_to_canonical.get(term, term)
            tokens.append(canonical)
            if canonical != term:
                normalizations.append(
                    {"source": term, "canonical": canonical}
                )

        if not tokens:
            compact = "".join(character for character in text if not character.isspace())
            tokens = [
                compact[index : index + 2]
                for index in range(max(0, len(compact) - 1))
            ]

        return tokens, normalizations

    def semantic_concepts(self, text: str) -> list[str]:
        """用词典覆盖率为离线编码器补充领域概念特征。"""
        text_characters = {
            character for character in text if not character.isspace()
        }
        concepts: list[str] = []
        for canonical, aliases in self.data.synonyms.items():
            for candidate in [canonical, *aliases]:
                characters = set(candidate)
                overlap = characters & text_characters
                if (
                    len(overlap) >= 3
                    and len(overlap) / max(len(characters), 1) >= 0.75
                ):
                    concepts.append(canonical)
                    break
        return concepts


class DenseEncoder:
    """OpenAI 兼容Embedding优先，未配置时使用确定性教学编码器。"""

    def __init__(self, tokenizer: QueryTokenizer, dimension: int = 192):
        self.tokenizer = tokenizer
        self.fallback_dimension = dimension
        self._cache: dict[str, list[float]] = {}
        self._forced_fallback = False
        self._active_provider: str | None = None
        self._expected_dimension: int | None = None

    def _desired_provider(self) -> str:
        return (
            model_runtime.embedding_provider_name
            if model_runtime.embedding_ready
            else FALLBACK_EMBEDDING_PROVIDER
        )

    def _sync_provider(self) -> str:
        desired = self._desired_provider()
        if self._active_provider and self._active_provider != desired:
            self._cache.clear()
            self._forced_fallback = False
            self._active_provider = None
            self._expected_dimension = None
        return desired

    @property
    def provider_name(self) -> str:
        desired = self._sync_provider()
        if self._forced_fallback:
            return FALLBACK_EMBEDDING_PROVIDER
        return self._active_provider or desired

    def set_expected_dimension(self, dimension: int) -> None:
        self._expected_dimension = dimension

    def _fallback_encode(self, text: str) -> list[float]:
        tokens, _ = self.tokenizer.tokenize(text)
        features = list(tokens)
        for concept in self.tokenizer.semantic_concepts(text):
            features.extend([f"concept:{concept}"] * 6)
        compact = "".join(character for character in text if not character.isspace())
        features.extend(
            compact[index : index + size]
            for size in (2, 3)
            for index in range(max(0, len(compact) - size + 1))
        )

        vector = [0.0] * self.fallback_dimension
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            for offset in range(0, 12, 3):
                index = int.from_bytes(digest[offset : offset + 2], "big")
                index %= self.fallback_dimension
                sign = 1.0 if digest[offset + 2] % 2 == 0 else -1.0
                vector[index] += sign / (1.0 + offset / 3)
        return _normalize(vector)

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        desired_provider = self._sync_provider()
        missing = [text for text in texts if text not in self._cache]
        if missing and (
            not self._forced_fallback
            and model_runtime.embedding_ready
        ):
            try:
                encoded: dict[str, list[float]] = {}
                for begin in range(0, len(missing), 64):
                    batch = missing[begin : begin + 64]
                    vectors = model_runtime.embed_many(batch)
                    for text, vector in zip(batch, vectors, strict=True):
                        encoded[text] = _normalize(vector)

                dimensions = {len(vector) for vector in encoded.values()}
                if len(dimensions) != 1:
                    raise RuntimeError("Embedding向量维度不一致")
                dimension = dimensions.pop()
                if (
                    self._expected_dimension is not None
                    and dimension != self._expected_dimension
                ):
                    raise RuntimeError(
                        "Embedding模型与已存储FAISS索引的维度不一致"
                    )
                self._cache.update(encoded)
                self._active_provider = desired_provider
                self._expected_dimension = dimension
            except Exception:
                if (
                    self._expected_dimension is not None
                    and self.provider_name
                    != FALLBACK_EMBEDDING_PROVIDER
                ):
                    raise RuntimeError(
                        "Embedding API不可用，不能用另一语义空间查询现有索引"
                    )
                self._forced_fallback = True
                self._active_provider = (
                    FALLBACK_EMBEDDING_PROVIDER
                )
                for text in missing:
                    self._cache[text] = self._fallback_encode(text)
        elif missing:
            self._active_provider = FALLBACK_EMBEDDING_PROVIDER
            for text in missing:
                self._cache[text] = self._fallback_encode(text)
        return [self._cache[text] for text in texts]

    def encode(self, text: str) -> list[float]:
        return self.encode_many([text])[0]


class DocumentRagService:
    def __init__(self, data: TeachingData):
        self.data = data
        self.tokenizer = QueryTokenizer(data)
        self.encoder = DenseEncoder(self.tokenizer)
        self.vector_index = DocumentFaissIndex(data, self.encoder)
        self.rows = data.tokenized_chunks
        self.chunk_by_id = {
            chunk["chunk_id"]: chunk for chunk in data.chunks
        }
        self.term_counts = [
            Counter(row["tokens"]) for row in self.rows
        ]
        self.doc_lengths = [
            sum(counter.values()) for counter in self.term_counts
        ]
        self.average_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.df = data.document_frequency
        self.n_documents = len(self.rows)

    def _query_terms(
        self, question: str
    ) -> tuple[list[str], list[dict[str, str]]]:
        return self.tokenizer.tokenize(question)

    def tfidf(
        self, question: str, top_k: int
    ) -> tuple[str, list[TraceStage], list[EvidenceItem], dict[str, Any]]:
        started = time.perf_counter()
        terms, normalizations = self._query_terms(question)
        vocabulary = sorted(self.data.token_vocabulary)
        vocabulary_index = {
            term: index for index, term in enumerate(vocabulary)
        }
        query_counts = Counter(
            term for term in terms if term in vocabulary_index
        )
        out_of_vocabulary = [
            term for term in terms if term not in vocabulary_index
        ]

        idf: dict[str, float] = {
            term: math.log(
                (self.n_documents + 1) / (self.df.get(term, 0) + 1)
            )
            + 1
            for term in vocabulary
        }
        query_raw = [
            query_counts.get(term, 0) * idf[term]
            for term in vocabulary
        ]
        query_raw_norm = _l2_norm(query_raw)
        query_vector = _normalize(query_raw)

        scored: list[dict[str, Any]] = []
        for index, counter in enumerate(self.term_counts):
            document_raw = [
                counter.get(term, 0) * idf[term] for term in vocabulary
            ]
            document_raw_norm = _l2_norm(document_raw)
            document_vector = _normalize(document_raw)
            score = _dot(query_vector, document_vector)
            if score > 0:
                chunk = self.chunk_by_id[self.rows[index]["chunk_id"]]
                products = [
                    {
                        "index": position,
                        "term": term,
                        "query": round(query_vector[position], 6),
                        "document": round(
                            document_vector[position],
                            6,
                        ),
                        "product": round(
                            query_vector[position]
                            * document_vector[position],
                            6,
                        ),
                    }
                    for term, position in vocabulary_index.items()
                    if query_vector[position] != 0
                ]
                scored.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": chunk["doc_id"],
                        "text": chunk["text"],
                        "score": score,
                        "raw_norm": document_raw_norm,
                        "vector": document_vector,
                        "products": products,
                    }
                )
        scored.sort(key=lambda item: item["score"], reverse=True)
        top = scored[:top_k]

        term_rows = [
            {
                "index": vocabulary_index[term],
                "term": term,
                "tf": query_counts[term],
                "df": self.df.get(term, 0),
                "idf": round(idf[term], 4),
                "weight": round(
                    query_raw[vocabulary_index[term]],
                    4,
                ),
                "normalized": round(
                    query_vector[vocabulary_index[term]],
                    4,
                ),
            }
            for term in dict.fromkeys(terms)
            if term in vocabulary_index
        ]
        elapsed = int((time.perf_counter() - started) * 1000)
        vocabulary_payload = [
            {
                "index": index,
                "term": term,
                "df": self.df.get(term, 0),
                "idf": round(idf[term], 4),
                "active": query_counts.get(term, 0) > 0,
            }
            for index, term in enumerate(vocabulary)
        ]
        comparison_payload = [
            {
                "id": item["chunk_id"],
                "title": item["doc_id"],
                "score": round(item["score"], 6),
                "query_norm": round(_l2_norm(query_vector), 6),
                "document_norm": round(
                    _l2_norm(item["vector"]),
                    6,
                ),
                "document_raw_norm": round(
                    item["raw_norm"],
                    6,
                ),
                "query_vector": [
                    round(value, 6) for value in query_vector
                ],
                "document_vector": [
                    round(value, 6) for value in item["vector"]
                ],
                "products": item["products"],
                "numerator": round(
                    sum(row["product"] for row in item["products"]),
                    6,
                ),
                "denominator": round(
                    _l2_norm(query_vector)
                    * _l2_norm(item["vector"]),
                    6,
                ),
                "excerpt": item["text"][:220],
            }
            for item in top
        ]
        stages = [
            TraceStage(
                id="tokenize",
                title="查询文本进入词项系统",
                kind="tokens",
                duration_ms=1,
                data={
                    "question": question,
                    "tokens": terms,
                    "normalizations": normalizations,
                    "document_count": self.n_documents,
                    "out_of_vocabulary": out_of_vocabulary,
                },
            ),
            TraceStage(
                id="tfidf-vocabulary",
                title="投影到语料词表坐标",
                kind="vocabulary-space",
                duration_ms=1,
                description=(
                    "向量维度由整个文档语料的固定词表决定；"
                    "查询命中的词只激活对应坐标，其余坐标为0。"
                ),
                data={
                    "dimension_count": len(vocabulary),
                    "document_count": self.n_documents,
                    "vocabulary": vocabulary_payload,
                    "active_terms": term_rows,
                    "out_of_vocabulary": out_of_vocabulary,
                },
            ),
            TraceStage(
                id="tfidf-vector",
                title="逐项构造查询向量",
                kind="tfidf-build",
                duration_ms=1,
                description=(
                    "每个激活坐标依次计算TF、IDF和乘积，"
                    "最后用整条向量的L2范数完成归一化。"
                ),
                data={
                    "formula": (
                        "w(t,q)=tf(t,q)×"
                        "[ln((N+1)/(df(t)+1))+1]"
                    ),
                    "terms": term_rows,
                    "dimensions": len(vocabulary),
                    "raw_norm": round(query_raw_norm, 6),
                    "raw_vector": [
                        round(value, 6) for value in query_raw
                    ],
                    "normalized_vector": [
                        round(value, 4) for value in query_vector
                    ],
                },
            ),
            TraceStage(
                id="cosine",
                title="查询向量与文档逐维相乘",
                kind="cosine-workbench",
                duration_ms=max(1, elapsed - 2),
                description=(
                    "查询向量和每个文档向量在同一110维坐标系中"
                    "对齐；同一坐标相乘后求和得到点积，再除以两者范数。"
                ),
                data={
                    "formula": "cos(q,d) = (q·d) / (||q|| ||d||)",
                    "dimension_count": len(vocabulary),
                    "vocabulary": vocabulary,
                    "comparisons": comparison_payload,
                },
            ),
            TraceStage(
                id="tfidf-ranking",
                title="按余弦得分回收Top-K",
                kind="ranking",
                duration_ms=1,
                description=(
                    "分数只负责排序和召回，返回的文本块随后作为证据进入生成阶段。"
                ),
                data={
                    "similarity": "cosine",
                    "results": [
                        {
                            "id": item["chunk_id"],
                            "title": item["doc_id"],
                            "score": round(item["score"], 4),
                            "excerpt": item["text"][:180],
                        }
                        for item in top
                    ],
                },
            ),
        ]
        evidence = [
            EvidenceItem(
                id=item["chunk_id"],
                title=item["doc_id"],
                excerpt=item["text"],
                score=round(item["score"], 4),
            )
            for item in top
        ]
        answer = (
            "TF–IDF返回了与查询关键词权重最接近的教学片段。"
            "它适合展示词项贡献，但不能仅凭文本相似度完成病例精确计数。"
        )
        return answer, stages, evidence, {
            "latency_ms": elapsed,
            "dimensions": len(vocabulary),
            "non_zero_dimensions": len(term_rows),
            "corpus_size": self.n_documents,
        }

    def bm25(
        self, question: str, top_k: int
    ) -> tuple[str, list[TraceStage], list[EvidenceItem], dict[str, Any]]:
        started = time.perf_counter()
        terms, normalizations = self._query_terms(question)
        unique_terms = list(dict.fromkeys(terms))
        k1 = 1.5
        b = 0.75
        idf = {
            term: math.log(
                1
                + (
                    self.n_documents
                    - self.df.get(term, 0)
                    + 0.5
                )
                / (self.df.get(term, 0) + 0.5)
            )
            for term in unique_terms
        }

        scored: list[dict[str, Any]] = []
        for index, counter in enumerate(self.term_counts):
            length = self.doc_lengths[index]
            term_details: list[dict[str, Any]] = []
            length_ratio = length / self.average_length
            length_normalizer = 1 - b + b * length_ratio
            for term in unique_terms:
                frequency = counter.get(term, 0)
                denominator = frequency + k1 * length_normalizer
                saturation = (
                    frequency * (k1 + 1) / denominator
                    if denominator
                    else 0
                )
                contribution = (
                    idf[term] * saturation
                )
                term_details.append(
                    {
                        "term": term,
                        "tf": frequency,
                        "df": self.df.get(term, 0),
                        "idf": idf[term],
                        "length_ratio": length_ratio,
                        "length_normalizer": length_normalizer,
                        "denominator": denominator,
                        "saturation": saturation,
                        "contribution": contribution,
                    }
                )
            score = sum(
                item["contribution"] for item in term_details
            )
            if score > 0:
                chunk = self.chunk_by_id[self.rows[index]["chunk_id"]]
                scored.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": chunk["doc_id"],
                        "text": chunk["text"],
                        "score": score,
                        "term_details": term_details,
                        "length": length,
                    }
                )
        scored.sort(key=lambda item: item["score"], reverse=True)
        top = scored[:top_k]
        elapsed = int((time.perf_counter() - started) * 1000)
        corpus_terms = [
            {
                "term": term,
                "document_count": self.n_documents,
                "df": self.df.get(term, 0),
                "idf": round(idf[term], 6),
                "document_ratio": round(
                    self.df.get(term, 0) / self.n_documents,
                    6,
                ),
            }
            for term in unique_terms
        ]
        document_payload = [
            {
                "id": item["chunk_id"],
                "title": item["doc_id"],
                "length": item["length"],
                "average_length": round(self.average_length, 4),
                "length_ratio": round(
                    item["length"] / self.average_length,
                    6,
                ),
                "score": round(item["score"], 6),
                "terms": [
                    {
                        key: (
                            round(value, 6)
                            if isinstance(value, float)
                            else value
                        )
                        for key, value in detail.items()
                    }
                    for detail in item["term_details"]
                ],
                "excerpt": item["text"][:220],
            }
            for item in top
        ]

        stages = [
            TraceStage(
                id="tokenize",
                title="查询文本进入词项系统",
                kind="tokens",
                data={
                    "question": question,
                    "tokens": terms,
                    "normalizations": normalizations,
                },
            ),
            TraceStage(
                id="bm25-corpus-statistics",
                title="从语料统计词项区分度",
                kind="bm25-corpus",
                description=(
                    "先统计每个查询词出现在多少个文档中；"
                    "出现越少，IDF越高，区分能力越强。"
                ),
                data={
                    "document_count": self.n_documents,
                    "terms": corpus_terms,
                    "idf_formula": (
                        "ln(1+(N-df+0.5)/(df+0.5))"
                    ),
                    "parameters": {
                        "k1": k1,
                        "b": b,
                        "avgdl": round(self.average_length, 2),
                    },
                },
            ),
            TraceStage(
                id="bm25-document-scoring",
                title="在每个文档内部计算词项贡献",
                kind="bm25-document",
                duration_ms=elapsed,
                description=(
                    "对每个候选文档读取词频与文档长度；"
                    "词频经过饱和，长度相对平均值完成归一化，"
                    "再乘IDF得到该词贡献。"
                ),
                data={
                    "formula": (
                        "contribution=IDF×"
                        "tf(k1+1)/(tf+k1(1-b+b·dl/avgdl))"
                    ),
                    "terms": unique_terms,
                    "parameters": {
                        "k1": k1,
                        "b": b,
                        "avgdl": round(self.average_length, 4),
                    },
                    "documents": document_payload,
                },
            ),
            TraceStage(
                id="bm25-accumulate",
                title="贡献矩阵沿文档行求和",
                kind="bm25-accumulator",
                description=(
                    "BM25不是把查询向量和文档向量直接做余弦；"
                    "它把每个查询词对当前文档的贡献相加，得到最终得分。"
                ),
                data={
                    "terms": unique_terms,
                    "documents": [
                        {
                            "id": item["chunk_id"],
                            "title": item["doc_id"],
                            "score": round(item["score"], 4),
                            "length": item["length"],
                            "contributions": [
                                round(
                                    detail["contribution"],
                                    6,
                                )
                                for detail in item["term_details"]
                            ],
                            "excerpt": item["text"][:180],
                        }
                        for item in top
                    ],
                },
            ),
            TraceStage(
                id="bm25-ranking",
                title="按BM25总分回收Top-K",
                kind="ranking",
                duration_ms=1,
                data={
                    "similarity": "BM25 score",
                    "results": [
                        {
                            "id": item["chunk_id"],
                            "title": item["doc_id"],
                            "score": round(item["score"], 4),
                            "excerpt": item["text"][:180],
                        }
                        for item in top
                    ],
                },
            ),
        ]
        evidence = [
            EvidenceItem(
                id=item["chunk_id"],
                title=item["doc_id"],
                excerpt=item["text"],
                score=round(item["score"], 4),
            )
            for item in top
        ]
        answer = (
            "BM25按词项区分度、词频饱和和文档长度完成排序。"
            "相较TF–IDF，它通常更适合关键词检索，但仍不负责关系遍历或精确聚合。"
        )
        return answer, stages, evidence, {
            "latency_ms": elapsed,
            "query_terms": len(unique_terms),
            "corpus_size": self.n_documents,
        }

    def semantic(
        self, question: str, top_k: int
    ) -> tuple[str, list[TraceStage], list[EvidenceItem], dict[str, Any]]:
        started = time.perf_counter()
        matches, query_vector, index_metadata = (
            self.vector_index.search(question, top_k)
        )
        top = [
            {
                "chunk": self.chunk_by_id[item["chunk_id"]],
                "score": item["score"],
                "position": item["position"],
                "vector": item["vector"],
            }
            for item in matches
        ]
        elapsed = int((time.perf_counter() - started) * 1000)
        comparison_payload = [
            {
                "id": item["chunk"]["chunk_id"],
                "title": item["chunk"]["doc_id"],
                "score": round(item["score"], 6),
                "position": item["position"],
                "query_vector": [
                    round(value, 6) for value in query_vector
                ],
                "document_vector": [
                    round(value, 6) for value in item["vector"]
                ],
                "groups": _group_vector_products(
                    query_vector,
                    item["vector"],
                ),
                "query_norm": round(_l2_norm(query_vector), 6),
                "document_norm": round(
                    _l2_norm(item["vector"]),
                    6,
                ),
                "projection": _project_vector_3d(item["vector"]),
                "excerpt": item["chunk"]["text"][:220],
            }
            for item in top
        ]
        stages = [
            TraceStage(
                id="semantic-input",
                title="语义输入",
                kind="text",
                data={
                    "text": question,
                    "provider": self.encoder.provider_name,
                    "normalization": "L2",
                },
            ),
            TraceStage(
                id="faiss-index",
                title="文档向量索引",
                kind="vector-index",
                description=(
                    "文本块已离线编码并持久化；运行时只编码查询，"
                    "再由FAISS返回Top-K行号。"
                ),
                data={
                    "input_file": "semantic_embedding_input.jsonl",
                    "index_file": "document_embeddings.faiss",
                    "metadata_file": (
                        "document_embeddings.meta.json"
                    ),
                    "index_type": index_metadata["index_type"],
                    "search_mode": index_metadata["search_mode"],
                    "similarity": index_metadata["similarity"],
                    "provider": index_metadata["provider"],
                    "dimensions": index_metadata["dimensions"],
                    "vector_count": index_metadata["vector_count"],
                    "load_source": index_metadata["load_source"],
                },
            ),
            TraceStage(
                id="dense-vector",
                title="形成完整稠密查询向量",
                kind="dense-vector",
                description=(
                    "所有维度共同承载语义，单个维度通常没有固定词义；"
                    "向量完成L2归一化后进入同一语义空间。"
                ),
                data={
                    "dimensions": len(query_vector),
                    "values": [
                        round(value, 5) for value in query_vector
                    ],
                    "norm": round(_l2_norm(query_vector), 5),
                    "projection": _project_vector_3d(query_vector),
                },
            ),
            TraceStage(
                id="faiss-space",
                title="查询向量进入FAISS语义空间",
                kind="vector-space",
                duration_ms=elapsed,
                data={
                    "query": {
                        "id": "QUERY",
                        "projection": _project_vector_3d(
                            query_vector
                        ),
                    },
                    "points": [
                        {
                            "id": item["id"],
                            "title": item["title"],
                            "score": item["score"],
                            "projection": item["projection"],
                        }
                        for item in comparison_payload
                    ],
                    "vector_count": index_metadata["vector_count"],
                    "dimensions": len(query_vector),
                    "projection_note": (
                        "空间坐标仅用于展示高维邻近关系；"
                        "精确排名仍由完整向量内积计算。"
                    ),
                },
            ),
            TraceStage(
                id="dense-similarity",
                title="完整向量逐维相乘并累加",
                kind="dense-similarity",
                description=(
                    "192个维度逐项相乘；前端按连续维度分组展示，"
                    "所有分组乘积之和就是归一化内积。"
                ),
                data={
                    "similarity": (
                        "L2归一化后的内积，等价于余弦相似度"
                    ),
                    "dimensions": len(query_vector),
                    "comparisons": comparison_payload,
                },
            ),
            TraceStage(
                id="faiss-ranking",
                title="FAISS返回Top-K行号与分数",
                kind="ranking",
                duration_ms=1,
                data={
                    "similarity": (
                        "L2归一化后的内积，等价于余弦相似度"
                    ),
                    "results": [
                        {
                            "id": item["chunk"]["chunk_id"],
                            "title": item["chunk"]["doc_id"],
                            "score": round(item["score"], 4),
                            "faiss_position": item["position"],
                            "excerpt": item["chunk"]["text"][:180],
                        }
                        for item in top
                    ],
                },
            ),
        ]
        evidence = [
            EvidenceItem(
                id=item["chunk"]["chunk_id"],
                title=item["chunk"]["doc_id"],
                excerpt=item["chunk"]["text"],
                score=round(item["score"], 4),
            )
            for item in top
        ]
        answer = (
            "FAISS返回了与问题整体含义最接近的文本块。"
            "文档向量已持久化，在线阶段只生成查询向量并执行Top-K检索。"
        )
        return answer, stages, evidence, {
            "latency_ms": elapsed,
            "dimensions": len(query_vector),
            "provider": self.encoder.provider_name,
            "index_type": index_metadata["index_type"],
            "vector_count": index_metadata["vector_count"],
        }
