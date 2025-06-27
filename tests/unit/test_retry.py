# Tests for the retry_on_exception decorator

import pytest

from src.tubeatlas.utils.retry import retry_on_exception

# ---------------------------------------------------------------------------
# Synchronous helper ---------------------------------------------------------
# ---------------------------------------------------------------------------

_sync_state = {"attempts": 0}


@retry_on_exception((ValueError,), max_retries=2, initial_wait=0.0, max_wait=0.0)
def flaky_sync(fail_times: int):
    """Function that raises ValueError *fail_times* before succeeding."""
    if _sync_state["attempts"] < fail_times:
        _sync_state["attempts"] += 1
        raise ValueError("boom")
    return "ok"


def test_retry_sync_success():
    _sync_state["attempts"] = 0
    assert flaky_sync(1) == "ok"  # succeeds on 2nd try


def test_retry_sync_exceed_retries():
    _sync_state["attempts"] = 0
    with pytest.raises(ValueError):
        flaky_sync(3)  # would need 4 tries, but max_retries=2


# ---------------------------------------------------------------------------
# Asynchronous helper --------------------------------------------------------
# ---------------------------------------------------------------------------

_async_state = {"attempts": 0}


@retry_on_exception((ValueError,), max_retries=2, initial_wait=0.0, max_wait=0.0)
async def flaky_async(fail_times: int):
    """Async function that raises ValueError *fail_times* before succeeding."""
    if _async_state["attempts"] < fail_times:
        _async_state["attempts"] += 1
        raise ValueError("boom")
    return "ok"


@pytest.mark.asyncio
async def test_retry_async_success():
    _async_state["attempts"] = 0
    result = await flaky_async(1)
    assert result == "ok"


@pytest.mark.asyncio
async def test_retry_async_exceed_retries():
    _async_state["attempts"] = 0
    with pytest.raises(ValueError):
        await flaky_async(3)
