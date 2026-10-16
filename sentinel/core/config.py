"""
Sentinel Configuration — Pydantic Settings loaded from .env file.

All configurable parameters for the engine: LLM model, embedding model,
chunk sizes, infrastructure URLs, shard count, and more.
"""

from __future__ import annotations

from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Sentinel engine.

    Loads values from a `.env` file at project root, with environment
    variable overrides taking precedence.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # ── Embeddings ───────────────────────────────────────────────────
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 100

    # ── Chunking ─────────────────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 100

    # ── Qdrant (sharded) ─────────────────────────────────────────────
    qdrant_urls: list[str] = [
        "http://qdrant-0:6333",
        "http://qdrant-1:6334",
    ]
    qdrant_collection: str = "sentinel_docs"
    qdrant_virtual_nodes: int = 150
    qdrant_search_timeout: float = 3.0
    qdrant_top_k: int = 20

    # ── Redis ────────────────────────────────────────────────────────
    redis_url: str = "redis://redis:6379/0"
    redis_stream_crawl_jobs: str = "crawl_jobs"
    redis_stream_dlq: str = "crawl_dlq"
    redis_consumer_group: str = "sentinel_workers"
    redis_idempotency_ttl: int = 86400  # 24 hours

    # ── PostgreSQL ───────────────────────────────────────────────────
    postgres_url: str = "postgresql+asyncpg://sentinel:sentinel@postgres:5432/sentinel"
    postgres_sync_url: str = "postgresql://sentinel:sentinel@postgres:5432/sentinel"

    # ── Ingestion ────────────────────────────────────────────────────
    worker_count: int = 2
    crawl_interval_hours: float = 6.0
    max_retries: int = 3
    rate_limit_requests: int = 1
    rate_limit_period: float = 2.0  # seconds per request per domain
    http_timeout: float = 30.0

    # ── Retrieval / RAG ──────────────────────────────────────────────
    retriever_top_k: int = 20
    reranker_top_k: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rrf_k: int = 60

    # ── Change Detection ─────────────────────────────────────────────
    significance_threshold: float = 0.5
    keyword_triggers: list[str] = [
        "pricing",
        "new",
        "launch",
        "deprecat",
        "remov",
        "feature",
        "update",
        "breaking",
    ]

    # ── Circuit Breaker ──────────────────────────────────────────────
    cb_failure_threshold: int = 5
    cb_failure_rate: float = 0.5
    cb_window_seconds: float = 30.0
    cb_recovery_timeout: float = 30.0

    # ── Competitors config ───────────────────────────────────────────
    competitors_file: str = "competitors.yaml"

    # ── API ───────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["*"]

    @field_validator("qdrant_urls", "keyword_triggers", "cors_origins", mode="before")
    @classmethod
    def parse_comma_separated(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            # Try JSON first
            import json
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            # Fall back to comma-separated
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


settings = Settings()
