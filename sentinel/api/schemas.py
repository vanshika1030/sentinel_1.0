"""Pydantic request/response schemas for all API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Query ────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for the /api/query endpoint."""
    question: str = Field(..., description="Natural language question", min_length=1)
    competitor: str | None = Field(None, description="Filter by competitor name")
    since: datetime | None = Field(None, description="Only consider data after this date")
    until: datetime | None = Field(None, description="Only consider data before this date")


class SourceInfo(BaseModel):
    """A source document referenced in an answer."""
    url: str = ""
    title: str = ""
    snippet: str = ""
    relevance_score: float = 0.0


class ChangeInfo(BaseModel):
    """A detected change referenced in an answer."""
    url: str = ""
    diff_summary: str = ""
    significance: float = 0.0
    detected_at: datetime | None = None


class QueryResponse(BaseModel):
    """Response body for the /api/query endpoint."""
    answer: str
    sources: list[SourceInfo] = []
    changes: list[ChangeInfo] = []


# ── Competitors ──────────────────────────────────────────────────

class CompetitorURL(BaseModel):
    """A URL to monitor for a competitor."""
    url: str = Field(..., description="URL to monitor")
    source_type: str = Field(
        "website", description="Source type: website, github, rss"
    )


class CompetitorCreate(BaseModel):
    """Request body for adding a competitor."""
    name: str = Field(..., description="Competitor name", min_length=1)
    urls: list[CompetitorURL] = Field(..., description="URLs to monitor")


class CompetitorInfo(BaseModel):
    """Info about a configured competitor."""
    name: str
    urls: list[CompetitorURL] = []


class CompetitorListResponse(BaseModel):
    """Response for listing competitors."""
    competitors: list[CompetitorInfo] = []


class ChangeTimelineResponse(BaseModel):
    """Response for change timeline queries."""
    competitor: str
    changes: list[ChangeInfo] = []
    total: int = 0


# ── Ingest ───────────────────────────────────────────────────────

class IngestTriggerRequest(BaseModel):
    """Request body for triggering a crawl."""
    competitor_name: str | None = Field(
        None, description="Specific competitor to crawl (all if omitted)"
    )


class IngestTriggerResponse(BaseModel):
    """Response for crawl trigger."""
    message: str
    jobs_queued: int = 0


class IngestStatusResponse(BaseModel):
    """Response for ingest status."""
    queue_depth: int = 0
    dlq_size: int = 0
    workers_active: int = 0


# ── Health ───────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response for health check."""
    status: str = "healthy"
    components: dict[str, str] = {}
