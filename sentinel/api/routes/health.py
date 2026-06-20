"""Health route — system health checks."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from sentinel.api.schemas import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check system health of all components."""
    from sentinel.api.app import get_app_state

    try:
        state = get_app_state()
        health_checker = state.get("health_checker")

        if health_checker is None:
            return HealthResponse(
                status="unknown",
                components={"note": "health checker not initialized"},
            )

        result = await health_checker.full_check()
        return HealthResponse(
            status=result["status"],
            components=result["components"],
        )
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        return HealthResponse(
            status="unhealthy",
            components={"error": str(exc)},
        )
