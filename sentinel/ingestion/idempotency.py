"""Idempotency guard — deduplicates content using SHA-256 hashes.

Stores content hashes in Redis with a configurable TTL to prevent
re-processing of unchanged content within the dedup window.
"""

from __future__ import annotations

import hashlib
import logging

import redis.asyncio as aioredis

from sentinel.core.config import settings

logger = logging.getLogger(__name__)


class IdempotencyGuard:
    """Deduplicates content processing using SHA-256 hashes stored in Redis."""

    def __init__(
        self,
        redis_client: aioredis.Redis | None = None,
        ttl: int | None = None,
    ) -> None:
        """Initialize the idempotency guard.

        Args:
            redis_client: Async Redis client.
            ttl: Time-to-live for hash keys in seconds.
        """
        self._redis = redis_client
        self._ttl = ttl or settings.redis_idempotency_ttl

    async def _get_redis(self) -> aioredis.Redis:
        """Lazily get or create the Redis client."""
        if self._redis is None:
            self._redis = aioredis.from_url(
                settings.redis_url, decode_responses=True
            )
        return self._redis

    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute SHA-256 hash of content.

        Args:
            content: The text content to hash.

        Returns:
            Hex digest of the SHA-256 hash.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def is_duplicate(self, url: str, content: str) -> bool:
        """Check if content has already been processed.

        Args:
            url: The source URL.
            content: The fetched content text.

        Returns:
            True if this exact content was already processed recently.
        """
        r = await self._get_redis()
        content_hash = self.compute_hash(content)
        key = f"sentinel:idem:{url}:{content_hash}"

        exists = await r.exists(key)
        if exists:
            logger.debug("Duplicate content detected for %s", url)
            return True
        return False

    async def mark_processed(self, url: str, content: str) -> str:
        """Mark content as processed by storing its hash.

        Args:
            url: The source URL.
            content: The fetched content text.

        Returns:
            The content hash that was stored.
        """
        r = await self._get_redis()
        content_hash = self.compute_hash(content)
        key = f"sentinel:idem:{url}:{content_hash}"

        await r.setex(key, self._ttl, "1")
        logger.debug("Marked content as processed: %s", url)
        return content_hash
