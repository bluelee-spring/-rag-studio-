from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
load_dotenv(PROJECT_ROOT / "backend" / ".env")


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = "RAG 方法实验台"
    app_version: str = "0.7.0"
    data_dir: Path = DATA_DIR
    runtime_dir: Path = RUNTIME_DIR

    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7688")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    fuseki_query_url: str = os.getenv("FUSEKI_QUERY_URL", "")

    llm_api_base: str = os.getenv("LLM_API_BASE", "").rstrip("/")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "")
    enable_llm_planner: bool = _bool_env("ENABLE_LLM_PLANNER", False)
    enable_llm_answer: bool = _bool_env("ENABLE_LLM_ANSWER", True)

    cors_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        ).split(",")
        if origin.strip()
    )


settings = Settings()
