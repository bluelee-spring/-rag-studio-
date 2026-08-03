from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import settings


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@dataclass
class TeachingData:
    chunks: list[dict[str, Any]]
    embedding_inputs: list[dict[str, Any]]
    tokenized_chunks: list[dict[str, Any]]
    synonyms: dict[str, list[str]]
    graph_nodes: dict[str, dict[str, str]]
    graph_edges: list[dict[str, str]]
    sqlite_path: Path
    ontology_path: Path
    instances_path: Path
    shapes_path: Path

    @classmethod
    def load(cls) -> "TeachingData":
        data_dir = settings.data_dir
        documents_dir = data_dir / "documents"
        graph_dir = data_dir / "graph"

        with (documents_dir / "synonyms.json").open(
            "r", encoding="utf-8-sig"
        ) as handle:
            synonyms = json.load(handle)

        with (graph_dir / "nodes.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            nodes = {row["id"]: row for row in csv.DictReader(handle)}

        with (graph_dir / "edges.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            edges = list(csv.DictReader(handle))

        return cls(
            chunks=read_jsonl(documents_dir / "chunks.jsonl"),
            embedding_inputs=read_jsonl(
                documents_dir / "semantic_embedding_input.jsonl"
            ),
            tokenized_chunks=read_jsonl(
                documents_dir / "tokenized_chunks.jsonl"
            ),
            synonyms=synonyms,
            graph_nodes=nodes,
            graph_edges=edges,
            sqlite_path=data_dir
            / "relational"
            / "soybean_cases.sqlite3",
            ontology_path=graph_dir / "ontology.ttl",
            instances_path=graph_dir / "instances.ttl",
            shapes_path=graph_dir / "shapes.ttl",
        )

    def sqlite(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @property
    def document_frequency(self) -> Counter[str]:
        frequency: Counter[str] = Counter()
        for row in self.tokenized_chunks:
            frequency.update(set(row["tokens"]))
        return frequency

    @property
    def token_vocabulary(self) -> list[str]:
        terms: set[str] = set()
        for row in self.tokenized_chunks:
            terms.update(row["tokens"])
        for canonical, aliases in self.synonyms.items():
            terms.add(canonical)
            terms.update(aliases)
        return sorted(terms, key=lambda item: (-len(item), item))

    @property
    def edge_lookup(self) -> dict[str, list[dict[str, str]]]:
        lookup: dict[str, list[dict[str, str]]] = defaultdict(list)
        for edge in self.graph_edges:
            lookup[edge["relation"]].append(edge)
        return lookup


teaching_data = TeachingData.load()
