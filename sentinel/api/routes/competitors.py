"""Competitors routes — manage monitored competitors."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Query

from sentinel.api.schemas import (
    ChangeInfo,
    ChangeTimelineResponse,
    CompetitorCreate,
    CompetitorInfo,
    CompetitorListResponse,
    CompetitorURL,
)
from sentinel.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/competitors", tags=["competitors"])


def _load_competitors() -> list[dict]:
    """Load competitors from YAML config."""
    path = Path(settings.competitors_file)
    if not path.exists():
        return []
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("competitors", []) if data else []


def _save_competitors(competitors: list[dict]) -> None:
    """Save competitors to YAML config."""
    path = Path(settings.competitors_file)
    with open(path, "w") as f:
        yaml.dump({"competitors": competitors}, f, default_flow_style=False)


@router.get("", response_model=CompetitorListResponse)
async def list_competitors() -> CompetitorListResponse:
    """List all configured competitors."""
    competitors = _load_competitors()
    return CompetitorListResponse(
        competitors=[
            CompetitorInfo(
                name=c["name"],
                urls=[
                    CompetitorURL(url=u["url"], source_type=u["source_type"])
                    for u in c.get("urls", [])
                ],
            )
            for c in competitors
        ]
    )


@router.post("", response_model=CompetitorInfo, status_code=201)
async def add_competitor(request: CompetitorCreate) -> CompetitorInfo:
    """Add a new competitor to monitor."""
    competitors = _load_competitors()

    # Check for duplicates
    if any(c["name"] == request.name for c in competitors):
        raise HTTPException(
            status_code=409,
            detail=f"Competitor '{request.name}' already exists",
        )

    new_competitor = {
        "name": request.name,
        "urls": [
            {"url": u.url, "source_type": u.source_type}
            for u in request.urls
        ],
    }
    competitors.append(new_competitor)
    _save_competitors(competitors)

    logger.info("Added competitor: %s", request.name)
    return CompetitorInfo(
        name=request.name,
        urls=request.urls,
    )


@router.get("/{name}/changes", response_model=ChangeTimelineResponse)
async def get_competitor_changes(
    name: str,
    since: datetime | None = Query(None),
    until: datetime | None = Query(None),
    significant_only: bool = Query(False),
) -> ChangeTimelineResponse:
    """Get the change timeline for a specific competitor."""
    from sentinel.api.app import get_app_state

    try:
        state = get_app_state()
        timeline_store = state["timeline_store"]

        changes = await timeline_store.get_changes(
            competitor_name=name,
            since=since,
            until=until,
            significant_only=significant_only,
        )

        return ChangeTimelineResponse(
            competitor=name,
            changes=[
                ChangeInfo(
                    url=c.url,
                    diff_summary=c.diff_text[:200] if c.diff_text else "",
                    significance=c.significance_score,
                    detected_at=c.detected_at,
                )
                for c in changes
            ],
            total=len(changes),
        )
    except Exception as exc:
        logger.error("Failed to get changes for %s: %s", name, exc)
        raise HTTPException(status_code=500, detail=str(exc))
