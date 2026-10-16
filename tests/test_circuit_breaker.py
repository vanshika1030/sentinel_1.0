"""Tests for the circuit breaker state machine."""

import asyncio
import time

import pytest

from sentinel.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    circuit_breaker,
    get_breaker,
)


class TestCircuitBreaker:
    """Test suite for CircuitBreaker."""

    @pytest.fixture
    def breaker(self) -> CircuitBreaker:
        """Create a circuit breaker with fast timeouts for testing."""
        return CircuitBreaker(
            name="test",
            failure_threshold=3,
            failure_rate=0.5,
            window_seconds=5.0,
            recovery_timeout=0.5,  # Fast recovery for tests
        )

    def test_initial_state_is_closed(self, breaker: CircuitBreaker) -> None:
        """Circuit should start in CLOSED state."""
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_success_keeps_closed(self, breaker: CircuitBreaker) -> None:
        """Successful calls should keep the circuit CLOSED."""
        async def success():
            return "ok"

        result = await breaker.call(success)
        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_consecutive_failures_open_circuit(self, breaker: CircuitBreaker) -> None:
        """Consecutive failures exceeding threshold should OPEN the circuit."""
        async def failing():
            raise ConnectionError("service down")

        for _ in range(3):
            with pytest.raises(ConnectionError):
                await breaker.call(failing)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_calls(self, breaker: CircuitBreaker) -> None:
        """OPEN circuit should raise CircuitOpenError immediately."""
        async def failing():
            raise ConnectionError("service down")

        # Trip the breaker
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await breaker.call(failing)

        # Now calls should be rejected
        with pytest.raises(CircuitOpenError) as exc_info:
            await breaker.call(failing)

        assert exc_info.value.name == "test"
        assert exc_info.value.retry_after >= 0

    @pytest.mark.asyncio
    async def test_open_to_half_open_after_timeout(self, breaker: CircuitBreaker) -> None:
        """After recovery timeout, circuit should transition to HALF_OPEN."""
        async def failing():
            raise ConnectionError("service down")

        # Trip the breaker
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await breaker.call(failing)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.6)

        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_success_closes_circuit(self, breaker: CircuitBreaker) -> None:
        """Successful call in HALF_OPEN should close the circuit."""
        async def failing():
            raise ConnectionError("service down")

        async def success():
            return "recovered"

        # Trip the breaker
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await breaker.call(failing)

        # Wait for recovery timeout
        await asyncio.sleep(0.6)

        # Test call succeeds
        result = await breaker.call(success)
        assert result == "recovered"
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self, breaker: CircuitBreaker) -> None:
        """Failed call in HALF_OPEN should reopen the circuit."""
        async def failing():
            raise ConnectionError("service down")

        # Trip the breaker
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await breaker.call(failing)

        # Wait for recovery timeout
        await asyncio.sleep(0.6)
        assert breaker.state == CircuitState.HALF_OPEN

        # Test call fails
        with pytest.raises(ConnectionError):
            await breaker.call(failing)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self) -> None:
        """A success should reset the consecutive failure counter."""
        # Use a dedicated breaker with high failure_rate so only consecutive
        # threshold matters (not the sliding window rate).
        b = CircuitBreaker(
            name="test_reset",
            failure_threshold=3,
            failure_rate=0.99,
            window_seconds=5.0,
            recovery_timeout=0.5,
        )

        async def failing():
            raise ConnectionError("oops")

        async def success():
            return "ok"

        # 2 failures (under threshold of 3)
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await b.call(failing)

        # Success resets counter
        await b.call(success)

        # 2 more failures should not open (counter reset)
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await b.call(failing)

        assert b.state == CircuitState.CLOSED

    def test_manual_reset(self, breaker: CircuitBreaker) -> None:
        """Manual reset should force circuit to CLOSED."""
        breaker._state = CircuitState.OPEN
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_decorator_usage(self) -> None:
        """Circuit breaker decorator should wrap functions correctly."""
        call_count = 0

        @circuit_breaker(name="test_decorator", failure_threshold=2, recovery_timeout=0.3)
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("fail")
            return "success"

        # First two calls fail
        with pytest.raises(ValueError):
            await flaky_func()
        with pytest.raises(ValueError):
            await flaky_func()

        # Circuit should now be open
        with pytest.raises(CircuitOpenError):
            await flaky_func()

    @pytest.mark.asyncio
    async def test_get_breaker_reuses_instance(self) -> None:
        """get_breaker should return the same instance for the same name."""
        b1 = get_breaker("shared")
        b2 = get_breaker("shared")
        assert b1 is b2

    @pytest.mark.asyncio
    async def test_sync_function_support(self, breaker: CircuitBreaker) -> None:
        """Circuit breaker should also work with sync functions."""
        def sync_func():
            return 42

        result = await breaker.call(sync_func)
        assert result == 42
