"""Job consumer — reads crawl jobs from Redis Stream via consumer groups.

Each worker reads from the sentinel_workers consumer group, processes
crawl jobs, and ACKs on success. Failed messages are retried up to 3
times before being moved to the dead-letter queue.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any, Callable, Awaitable

import redis.asyncio as aioredis

from sentinel.core.config import settings

logger = logging.getLogger(__name__)


class JobConsumer:
    """Consumes crawl jobs from a Redis Stream consumer group."""

    def __init__(
        self,
        consumer_name: str,
        handler: Callable[[dict[str, Any]], Awaitable[None]],
        redis_client: aioredis.Redis | None = None,
    ) -> None:
        """Initialize the consumer.

        Args:
            consumer_name: Unique name for this consumer within the group.
            handler: Async callback to process each job message.
            redis_client: Async Redis client.
        """
        self._consumer_name = consumer_name
        self._handler = handler
        self._redis = redis_client
        self._running = False

    async def _get_redis(self) -> aioredis.Redis:
        """Lazily get or create the Redis client."""
        if self._redis is None:
            self._redis = aioredis.from_url(
                settings.redis_url, decode_responses=True
            )
        return self._redis

    async def _ensure_group(self) -> None:
        """Create the consumer group if it doesn't exist."""
        r = await self._get_redis()
        try:
            await r.xgroup_create(
                settings.redis_stream_crawl_jobs,
                settings.redis_consumer_group,
                id="0",
                mkstream=True,
            )
            logger.info("Created consumer group '%s'", settings.redis_consumer_group)
        except aioredis.ResponseError as exc:
            if "BUSYGROUP" in str(exc):
                pass  # Group already exists
            else:
                raise

    async def _move_to_dlq(self, message_id: str, data: dict[str, Any]) -> None:
        """Move a failed message to the dead-letter queue.

        Args:
            message_id: The original message ID.
            data: The message payload.
        """
        r = await self._get_redis()
        dlq_data = {**data, "original_id": message_id, "reason": "max_retries"}
        await r.xadd(settings.redis_stream_dlq, dlq_data)
        await r.xack(
            settings.redis_stream_crawl_jobs,
            settings.redis_consumer_group,
            message_id,
        )
        logger.warning("Moved message %s to DLQ", message_id)

    async def _process_message(
        self, message_id: str, data: dict[str, Any]
    ) -> None:
        """Process a single message with retry tracking.

        Args:
            message_id: Redis Stream message ID.
            data: Message payload.
        """
        r = await self._get_redis()
        retry_key = f"sentinel:retry:{message_id}"

        for attempt in range(1, settings.max_retries + 1):
            try:
                await self._handler(data)
                await r.xack(
                    settings.redis_stream_crawl_jobs,
                    settings.redis_consumer_group,
                    message_id,
                )
                logger.debug("ACKed message %s", message_id)
                return
            except Exception as exc:
                logger.error(
                    "Attempt %d/%d failed for message %s: %s",
                    attempt,
                    settings.max_retries,
                    message_id,
                    exc,
                )
                if attempt >= settings.max_retries:
                    await self._move_to_dlq(message_id, data)
                    return
                await asyncio.sleep(1.0 * attempt)  # Simple backoff

    async def _recover_pending(self) -> None:
        """Recover pending messages via XAUTOCLAIM."""
        r = await self._get_redis()
        try:
            # Claim messages idle for > 60 seconds
            result = await r.xautoclaim(
                settings.redis_stream_crawl_jobs,
                settings.redis_consumer_group,
                self._consumer_name,
                min_idle_time=60000,  # 60 seconds
                start_id="0-0",
            )
            if result and len(result) > 1 and result[1]:
                for msg_id, data in result[1]:
                    logger.info("Recovered pending message %s", msg_id)
                    await self._process_message(msg_id, data)
        except Exception as exc:
            logger.debug("Pending recovery: %s", exc)

    async def run(self) -> None:
        """Run the consumer loop."""
        await self._ensure_group()
        self._running = True
        r = await self._get_redis()

        logger.info("Consumer '%s' started", self._consumer_name)

        while self._running:
            try:
                # Recover any abandoned pending messages
                await self._recover_pending()

                # Read new messages
                messages = await r.xreadgroup(
                    groupname=settings.redis_consumer_group,
                    consumername=self._consumer_name,
                    streams={settings.redis_stream_crawl_jobs: ">"},
                    count=1,
                    block=5000,  # Block for 5 seconds
                )

                if messages:
                    for stream_name, stream_messages in messages:
                        for msg_id, data in stream_messages:
                            await self._process_message(msg_id, data)

            except asyncio.CancelledError:
                logger.info("Consumer '%s' cancelled", self._consumer_name)
                break
            except Exception as exc:
                logger.error("Consumer error: %s", exc)
                await asyncio.sleep(2.0)

    def stop(self) -> None:
        """Signal the consumer to stop."""
        self._running = False
