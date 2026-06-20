"""Worker pool — manages N async consumer workers.

Spawns configurable workers that each run the consumer loop,
processing crawl jobs concurrently.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Awaitable

import redis.asyncio as aioredis

from sentinel.core.config import settings
from sentinel.ingestion.consumer import JobConsumer

logger = logging.getLogger(__name__)


class WorkerPool:
    """Manages a pool of async consumer workers."""

    def __init__(
        self,
        handler: Callable[[dict[str, Any]], Awaitable[None]],
        worker_count: int | None = None,
        redis_client: aioredis.Redis | None = None,
    ) -> None:
        """Initialize the worker pool.

        Args:
            handler: Async callback to process each crawl job.
            worker_count: Number of concurrent workers. Defaults to config.
            redis_client: Shared async Redis client.
        """
        self._handler = handler
        self._worker_count = worker_count or settings.worker_count
        self._redis = redis_client
        self._consumers: list[JobConsumer] = []
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Start all consumer workers."""
        if self._redis is None:
            self._redis = aioredis.from_url(
                settings.redis_url, decode_responses=True
            )

        logger.info("Starting %d workers", self._worker_count)

        for i in range(self._worker_count):
            consumer = JobConsumer(
                consumer_name=f"worker-{i}",
                handler=self._handler,
                redis_client=self._redis,
            )
            self._consumers.append(consumer)
            task = asyncio.create_task(consumer.run(), name=f"worker-{i}")
            self._tasks.append(task)

        logger.info("All %d workers started", self._worker_count)

    async def stop(self) -> None:
        """Stop all consumer workers gracefully."""
        logger.info("Stopping %d workers", len(self._consumers))
        for consumer in self._consumers:
            consumer.stop()

        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._consumers.clear()
        self._tasks.clear()
        logger.info("All workers stopped")

    @property
    def active_workers(self) -> int:
        """Return the number of active (not done) worker tasks."""
        return sum(1 for t in self._tasks if not t.done())

    async def run_forever(self) -> None:
        """Start workers and block until interrupted."""
        await self.start()
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            await self.stop()
