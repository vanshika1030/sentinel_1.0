"""Retry with exponential backoff and jitter.

Decorator for async functions that retries on failure with
configurable attempts, base delay, and exponential backoff.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from typing import Any, Callable, Type

logger = logging.getLogger(__name__)


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """Decorator for async retry with exponential backoff and jitter.

    delay = base_delay * (2 ** attempt) + random(0, base_delay)

    Args:
        max_attempts: Maximum number of attempts (including first try).
        base_delay: Base delay in seconds for backoff calculation.
        retryable_exceptions: Tuple of exception types to retry on.

    Returns:
        Decorator function.

    Usage::

        @with_retry(max_attempts=3, base_delay=1.0)
        async def flaky_operation(): ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exception = exc
                    if attempt + 1 >= max_attempts:
                        logger.error(
                            "[retry] %s failed after %d attempts: %s",
                            func.__name__,
                            max_attempts,
                            exc,
                        )
                        raise

                    delay = base_delay * (2 ** attempt) + random.uniform(
                        0, base_delay
                    )
                    logger.warning(
                        "[retry] %s attempt %d/%d failed (%s), "
                        "retrying in %.2fs",
                        func.__name__,
                        attempt + 1,
                        max_attempts,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator
