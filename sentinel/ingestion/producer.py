"""Job producer — pushes crawl jobs to Redis Stream.

Creates structured job messages and adds them to the crawl_jobs stream
using XADD for reliable delivery to consumer workers.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import redis.asyncio as aioredis

from sentinel.core.config import settings

logger = logging.getLogger(__name__)


class JobProducer:
    """Pushes crawl jobs onto a Redis Stream."""

    def __init__(self, redis_client: aioredis.Redis | None = None) -> None:
        """Initialize the producer.

        Args:
            redis_client: Async Redis client. Creates one if not provided.
        """
        self._redis = redis_client

    async def _get_redis(self) -> aioredis.Redis:
        """Lazily get or create the Redis client."""
        if self._redis is None:
            self._redis = aioredis.from_url(
                settings.redis_url, decode_responses=True
            )
        return self._redis

    async def push_job(
        self,
        url: str,
        source_type: str,
        competitor_name: str,
        priority: int = 1,
    ) -> str:
        """Push a single crawl job to the Redis Stream.

        Args:
            url: The URL to crawl.
            source_type: Type of source (website, github, rss).
            competitor_name: Name of the competitor.
            priority: Job priority (1 = normal, higher = more urgent).

        Returns:
            The Redis Stream message ID.
        """
        r = await self._get_redis()
        job = {
            "url": url,
            "source_type": source_type,
            "competitor_name": competitor_name,
            "priority": str(priority),
            "created_at": str(time.time()),
        }
        msg_id = await r.xadd(settings.redis_stream_crawl_jobs, job)
        logger.info("Pushed job %s for %s (%s)", msg_id, url, competitor_name)
        return msg_id

    async def push_batch(
        self, jobs: list[dict[str, Any]]
    ) -> list[str]:
        """Push multiple crawl jobs.

        Args:
            jobs: List of dicts with keys: url, source_type, competitor_name.

        Returns:
            List of message IDs.
        """
        ids = []
        for job in jobs:
            msg_id = await self.push_job(
                url=job["url"],
                source_type=job["source_type"],
                competitor_name=job["competitor_name"],
                priority=job.get("priority", 1),
            )
            ids.append(msg_id)
        return ids

    async def get_queue_depth(self) -> int:
        """Return the number of messages in the crawl stream.

        Returns:
            Stream length.
        """
        r = await self._get_redis()
        return await r.xlen(settings.redis_stream_crawl_jobs)

    async def get_dlq_size(self) -> int:
        """Return the number of messages in the dead-letter queue.

        Returns:
            DLQ stream length.
        """
        r = await self._get_redis()
        try:
            return await r.xlen(settings.redis_stream_dlq)
        except Exception:
            return 0
