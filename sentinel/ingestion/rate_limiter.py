"""Token bucket rate limiter — per domain.

Simple token bucket implementation that enforces rate limits
on a per-domain basis to avoid overwhelming target servers.
"""

from __future__ import annotations

import asyncio
import time
from urllib.parse import urlparse

from sentinel.core.config import settings


class TokenBucket:
    """A simple token bucket rate limiter."""

    def __init__(self, rate: float, capacity: int) -> None:
        """Initialize the token bucket.

        Args:
            rate: Tokens added per second.
            capacity: Maximum number of tokens.
        """
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it."""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    self._capacity, self._tokens + elapsed * self._rate
                )
                self._last_refill = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

            # Wait for a token to become available
            wait_time = (1.0 - self._tokens) / self._rate
            await asyncio.sleep(wait_time)


class DomainRateLimiter:
    """Rate limiter that maintains per-domain token buckets."""

    def __init__(
        self,
        requests_per_period: int | None = None,
        period_seconds: float | None = None,
    ) -> None:
        """Initialize the domain rate limiter.

        Args:
            requests_per_period: Allowed requests per period.
            period_seconds: Period duration in seconds.
        """
        self._rate = (requests_per_period or settings.rate_limit_requests) / (
            period_seconds or settings.rate_limit_period
        )
        self._capacity = requests_per_period or settings.rate_limit_requests
        self._buckets: dict[str, TokenBucket] = {}

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract the domain from a URL.

        Args:
            url: The URL to parse.

        Returns:
            The domain name.
        """
        parsed = urlparse(url)
        return parsed.netloc or parsed.path

    def _get_bucket(self, domain: str) -> TokenBucket:
        """Get or create a token bucket for a domain.

        Args:
            domain: The domain name.

        Returns:
            The TokenBucket for this domain.
        """
        if domain not in self._buckets:
            self._buckets[domain] = TokenBucket(
                rate=self._rate, capacity=self._capacity
            )
        return self._buckets[domain]

    async def acquire(self, url: str) -> None:
        """Wait for rate limit clearance for a URL's domain.

        Args:
            url: The URL being requested.
        """
        domain = self._extract_domain(url)
        bucket = self._get_bucket(domain)
        await bucket.acquire()
