"""Circuit breaker — state machine for fault tolerance.

Implements the CLOSED → OPEN → HALF-OPEN pattern to prevent
cascading failures when downstream services are unhealthy.
Usable as a decorator.
"""

from __future__ import annotations

import asyncio
import enum
import functools
import logging
import time
from collections import deque
from typing import Any, Callable

from sentinel.core.config import settings

logger = logging.getLogger(__name__)


class CircuitOpenError(Exception):
    """Raised when a call is attempted on an open circuit."""

    def __init__(self, name: str, retry_after: float) -> None:
        self.name = name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit '{name}' is OPEN. Retry after {retry_after:.1f}s"
        )


class CircuitState(enum.Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker with sliding window failure tracking.

    State transitions:
        CLOSED → OPEN: when failure rate exceeds threshold OR
                        consecutive failures exceed threshold
        OPEN → HALF_OPEN: after recovery timeout elapses
        HALF_OPEN → CLOSED: if test call succeeds
        HALF_OPEN → OPEN: if test call fails
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int | None = None,
        failure_rate: float | None = None,
        window_seconds: float | None = None,
        recovery_timeout: float | None = None,
    ) -> None:
        """Initialize the circuit breaker.

        Args:
            name: Identifier for this circuit (e.g. 'qdrant_shard_0').
            failure_threshold: Consecutive failures before opening.
            failure_rate: Failure rate (0-1) to trigger opening.
            window_seconds: Sliding window duration for tracking.
            recovery_timeout: Seconds to wait in OPEN before HALF_OPEN.
        """
        self.name = name
        self._failure_threshold = failure_threshold or settings.cb_failure_threshold
        self._failure_rate = failure_rate or settings.cb_failure_rate
        self._window_seconds = window_seconds or settings.cb_window_seconds
        self._recovery_timeout = recovery_timeout or settings.cb_recovery_timeout

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float = 0.0
        self._lock = asyncio.Lock()

        # Sliding window: deque of (timestamp, success: bool)
        self._window: deque[tuple[float, bool]] = deque()

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._recovery_timeout:
                return CircuitState.HALF_OPEN
        return self._state

    def _prune_window(self) -> None:
        """Remove entries older than the sliding window duration."""
        cutoff = time.monotonic() - self._window_seconds
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    def _failure_rate_exceeded(self) -> bool:
        """Check if the failure rate within the window exceeds the threshold."""
        self._prune_window()
        if len(self._window) < 5:  # Need minimum sample
            return False
        failures = sum(1 for _, success in self._window if not success)
        rate = failures / len(self._window)
        return rate > self._failure_rate

    def _record_success(self) -> None:
        """Record a successful call."""
        self._window.append((time.monotonic(), True))
        self._consecutive_failures = 0

        if self._state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
            logger.info("Circuit '%s' → CLOSED (success)", self.name)
            self._state = CircuitState.CLOSED

    def _record_failure(self) -> None:
        """Record a failed call and potentially open the circuit."""
        self._window.append((time.monotonic(), False))
        self._consecutive_failures += 1

        should_open = (
            self._consecutive_failures >= self._failure_threshold
            or self._failure_rate_exceeded()
        )

        if self._state == CircuitState.HALF_OPEN or (
            self._state == CircuitState.CLOSED and should_open
        ):
            logger.warning(
                "Circuit '%s' → OPEN (consecutive=%d, rate_exceeded=%s)",
                self.name,
                self._consecutive_failures,
                self._failure_rate_exceeded(),
            )
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()

    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute a function through the circuit breaker.

        Args:
            func: The async function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The function's return value.

        Raises:
            CircuitOpenError: If the circuit is OPEN.
        """
        async with self._lock:
            current_state = self.state

            if current_state == CircuitState.OPEN:
                retry_after = self._recovery_timeout - (
                    time.monotonic() - self._opened_at
                )
                raise CircuitOpenError(self.name, max(0, retry_after))

            if current_state == CircuitState.HALF_OPEN:
                self._state = CircuitState.HALF_OPEN
                logger.info("Circuit '%s' HALF_OPEN — allowing test call", self.name)

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            async with self._lock:
                self._record_success()
            return result
        except Exception as exc:
            async with self._lock:
                self._record_failure()
            raise

    def reset(self) -> None:
        """Manually reset the circuit to CLOSED."""
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._window.clear()
        logger.info("Circuit '%s' manually reset to CLOSED", self.name)


# ── Registry of named circuit breakers ────────────────────────────
_breakers: dict[str, CircuitBreaker] = {}


def get_breaker(name: str, **kwargs: Any) -> CircuitBreaker:
    """Get or create a named circuit breaker.

    Args:
        name: Circuit breaker identifier.
        **kwargs: Configuration overrides.

    Returns:
        The CircuitBreaker instance for this name.
    """
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name=name, **kwargs)
    return _breakers[name]


def circuit_breaker(name: str, **kwargs: Any) -> Callable:
    """Decorator to wrap an async function with a circuit breaker.

    Args:
        name: Circuit breaker identifier.
        **kwargs: Configuration overrides.

    Returns:
        Decorator function.

    Usage::

        @circuit_breaker(name="qdrant_shard_0")
        async def search_shard(query): ...
    """
    def decorator(func: Callable) -> Callable:
        breaker = get_breaker(name, **kwargs)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kw: Any) -> Any:
            return await breaker.call(func, *args, **kw)

        wrapper.breaker = breaker  # type: ignore[attr-defined]
        return wrapper

    return decorator
