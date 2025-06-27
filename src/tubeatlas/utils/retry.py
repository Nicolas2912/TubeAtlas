import asyncio
import functools
import logging
import random
import time
from typing import Any, Callable, Tuple, Type

logger = logging.getLogger(__name__)


def retry_on_exception(  # noqa: C901
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    *,
    max_retries: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.25,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a decorator that retries the wrapped callable on given exceptions.

    The decorator supports both **sync** and **async** callables. Exponential back-off
    with full-jitter (*±jitter*) is applied between attempts.

    Args:
        exceptions: Exception classes that trigger a retry.
        max_retries: Maximum number of *additional* retry attempts (excluding the first).
        initial_wait: Initial delay in seconds before the first retry.
        max_wait: Upper bound for the delay between retries.
        backoff_factor: Multiplier applied to the wait time after each failed attempt.
        jitter: Fractional jitter (+/-) applied to the calculated wait time to avoid
            thundering-herd issues.

    Example::

        >>> @retry_on_exception((RuntimeError,), max_retries=5)
        ... async def fetch_remote():
        ...     ...

    """

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        is_coro = asyncio.iscoroutinefunction(func)

        async def _async_retry_wrapper(*args: Any, **kwargs: Any):  # type: ignore[name-defined]
            wait = initial_wait
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)  # type: ignore[arg-type]
                except exceptions as exc:
                    if attempt >= max_retries:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__name__,
                            attempt + 1,
                            exc,
                        )
                        raise
                    wait = _log_retry(func, attempt, exc, wait, jitter)
                    await asyncio.sleep(wait)
                    wait = min(wait * backoff_factor, max_wait)

        def _sync_retry_wrapper(*args: Any, **kwargs: Any):  # type: ignore[name-defined]
            wait = initial_wait
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt >= max_retries:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__name__,
                            attempt + 1,
                            exc,
                        )
                        raise
                    wait = _log_retry(func, attempt, exc, wait, jitter)
                    time.sleep(wait)
                    wait = min(wait * backoff_factor, max_wait)

        functools.update_wrapper(
            _async_retry_wrapper if is_coro else _sync_retry_wrapper, func
        )
        return _async_retry_wrapper if is_coro else _sync_retry_wrapper  # type: ignore[return-value]

    return _decorator


def _log_retry(
    func: Callable[..., Any],
    attempt: int,
    exc: Exception,
    wait: float,
    jitter: float,
) -> float:
    """Log retry attempt with jitter-aware wait time and return the actual wait."""
    jitter_factor = 1 + (random.random() * 2 - 1) * jitter
    actual_wait = max(0.0, wait * jitter_factor)
    logger.warning(
        "%s raised %s on attempt %d — retrying in %.2fs",
        func.__name__,
        exc.__class__.__name__,
        attempt + 1,
        actual_wait,
    )
    return actual_wait
