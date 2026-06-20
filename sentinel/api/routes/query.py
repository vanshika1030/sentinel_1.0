"""Query route — natural language questions over competitor data."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from sentinel.api.schemas import (
    ChangeInfo,
    QueryRequest,
    QueryResponse,
    SourceInfo,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_intelligence(request: QueryRequest) -> QueryResponse:
    """Ask a natural language question about competitor intelligence.

    Embeds the question, scatter-gathers across shards, applies hybrid
    retrieval with re-ranking, and generates an LLM answer with citations.
    """
    from sentinel.api.app import get_app_state

    try:
        state = get_app_state()
        embedder = state["embedder"]
        scatter_gather = state["scatter_gather"]
        retriever = state["retriever"]
        generator = state["generator"]
        timeline_store = state["timeline_store"]

        # Embed the question
        query_vector = await embedder.embed_query(request.question)

        # Scatter-gather across shards
        vector_results = await scatter_gather.search(
            query_vector=query_vector,
            top_k=20,
        )

        # Hybrid retrieval with BM25 + RRF + cross-encoder
        reranked = await retriever.retrieve(
            query=request.question,
            vector_results=vector_results,
        )

        # Generate answer
        answer = await generator.generate(
            question=request.question,
            documents=reranked,
        )

        # Get relevant changes
        changes = []
        try:
            change_records = await timeline_store.get_changes(
                competitor_name=request.competitor,
                since=request.since,
                until=request.until,
                significant_only=True,
                limit=10,
            )
            changes = [
                ChangeInfo(
                    url=c.url,
                    diff_summary=c.diff_text[:200] if c.diff_text else "",
                    significance=c.significance_score,
                    detected_at=c.detected_at,
                )
                for c in change_records
            ]
        except Exception as exc:
            logger.warning("Failed to fetch changes: %s", exc)

        return QueryResponse(
            answer=answer.answer,
            sources=[
                SourceInfo(**s) for s in answer.sources
            ],
            changes=changes,
        )

    except Exception as exc:
        logger.error("Query failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")
