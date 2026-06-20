"""Ingest routes — trigger crawls and check ingest status."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from sentinel.api.schemas import (
    IngestStatusResponse,
    IngestTriggerRequest,
    IngestTriggerResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ingest", tags=["ingest"])


@router.post("/trigger", response_model=IngestTriggerResponse)
async def trigger_crawl(request: IngestTriggerRequest) -> IngestTriggerResponse:
    """Manually trigger a crawl for one or all competitors."""
    from sentinel.api.app import get_app_state

    try:
        state = get_app_state()
        scheduler = state["scheduler"]
        producer = state["producer"]

        await scheduler.trigger_crawl(request.competitor_name)

        depth = await producer.get_queue_depth()
        target = request.competitor_name or "all competitors"
        return IngestTriggerResponse(
            message=f"Crawl triggered for {target}",
            jobs_queued=depth,
        )
    except Exception as exc:
        logger.error("Failed to trigger crawl: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/status", response_model=IngestStatusResponse)
async def ingest_status() -> IngestStatusResponse:
    """Return current ingestion pipeline status."""
    from sentinel.api.app import get_app_state

    try:
        state = get_app_state()
        producer = state["producer"]
        worker_pool = state.get("worker_pool")

        return IngestStatusResponse(
            queue_depth=await producer.get_queue_depth(),
            dlq_size=await producer.get_dlq_size(),
            workers_active=worker_pool.active_workers if worker_pool else 0,
        )
    except Exception as exc:
        logger.error("Failed to get ingest status: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
