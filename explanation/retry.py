"""
Exponential backoff retry for Anthropic API calls.

The SDK's built-in retry caps backoff at 8s, which isn't enough for 529
(overloaded) errors. This module provides a retry wrapper with longer
backoff windows specifically tuned for transient overload.

Schedule (with jitter): ~1s → 2s → 4s → 8s → 16s → 32s
Total max wait before giving up: ~63s
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TypeVar

from anthropic._exceptions import OverloadedError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Retry config
MAX_ATTEMPTS = 6
INITIAL_DELAY = 1.0  # seconds
MAX_DELAY = 32.0  # seconds
JITTER = 0.5  # ± fraction of delay


async def retry_overloaded(coro_factory, /) -> anthropic.types.Message:
    """
    Call coro_factory() repeatedly until it succeeds or attempts are exhausted.

    coro_factory must be a zero-arg callable that returns an awaitable
    (e.g. a lambda or functools.partial), since a coroutine can only be
    awaited once.

    Retries only on OverloadedError (529). All other exceptions propagate
    immediately.
    """
    last_exc = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return await coro_factory()
        except OverloadedError as exc:
            last_exc = exc
            if attempt == MAX_ATTEMPTS - 1:
                break
            delay = min(INITIAL_DELAY * (2 ** attempt), MAX_DELAY)
            delay *= 1.0 + JITTER * (2 * random.random() - 1)
            logger.warning(
                "Anthropic 529 overloaded (attempt %d/%d), retrying in %.1fs",
                attempt + 1,
                MAX_ATTEMPTS,
                delay,
            )
            await asyncio.sleep(delay)

    raise last_exc  # type: ignore[misc]
