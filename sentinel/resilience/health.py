"""System health checks — Redis, Qdrant shards, PostgreSQL.

Returns aggregated health status: healthy / degraded / unhealthy.
"""

from __future__ import annotations

import logging
from typing import Any

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.text import text

from sentinel.core.config import settings
from sentinel.sharding.shard_manager import ShardManager

logger = logging.getLogger(__name__)


class HealthChecker:
    """Checks health of all infrastructure components."""

    def __init__(
        self,
        redis_client: aioredis.Redis | None = None,
        shard_manager: ShardManager | None = None,
        db_engine: AsyncEngine | None = None,
    ) -> None:
        """Initialize the health checker.

        Args:
            redis_client: Async Redis client.
            shard_manager: Qdrant shard manager.
            db_engine: SQLAlchemy async engine.
        """
        self._redis = redis_client
        self._shard_manager = shard_manager
        self._db_engine = db_engine

    async def check_redis(self) -> dict[str, str]:
        """Check Redis connectivity.

        Returns:
            Dict with Redis health status.
        """
        if self._redis is None:
            return {"redis": "not_configured"}
        try:
            await self._redis.ping()
            return {"redis": "up"}
        except Exception as exc:
            logger.warning("Redis health check failed: %s", exc)
            return {"redis": "down"}

    async def check_qdrant(self) -> dict[str, str]:
        """Check all Qdrant shards.

        Returns:
            Dict mapping shard names to health status.
        """
        if self._shard_manager is None:
            return {"qdrant": "not_configured"}

        health = await self._shard_manager.health_check()
        result = {}
        for i, (url, is_healthy) in enumerate(health.items()):
            result[f"qdrant_{i}"] = "up" if is_healthy else "down"
        return result

    async def check_postgres(self) -> dict[str, str]:
        """Check PostgreSQL connectivity.

        Returns:
            Dict with PostgreSQL health status.
        """
        if self._db_engine is None:
            return {"postgres": "not_configured"}
        try:
            async with self._db_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            return {"postgres": "up"}
        except Exception as exc:
            logger.warning("PostgreSQL health check failed: %s", exc)
            return {"postgres": "down"}

    async def full_check(self) -> dict[str, Any]:
        """Run all health checks and return aggregated status.

        Returns:
            Dict with overall status and per-component status.
            status is one of: healthy, degraded, unhealthy.
        """
        components: dict[str, str] = {}
        components.update(await self.check_redis())
        components.update(await self.check_qdrant())
        components.update(await self.check_postgres())

        down_count = sum(1 for v in components.values() if v == "down")
        total = sum(1 for v in components.values() if v != "not_configured")

        if down_count == 0:
            status = "healthy"
        elif down_count < total:
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            "status": status,
            "components": components,
        }
