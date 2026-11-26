"""Small helpers: rate limiting, retries, text slugs, JSON-safe conversion."""

from __future__ import annotations

import math
import random
import re
import time
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")


def slugify(text: str) -> str:
    """'Qwen/Qwen2.5-3B-Instruct' -> 'qwen2.5-3b-instruct' (safe for file names)."""
    text = text.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9.]+", "-", text).strip("-")


class RateLimiter:
    """Keeps API calls under `per_minute` requests per minute (None = no limit)."""

    def __init__(self, per_minute: float | None):
        self.min_interval = 60.0 / per_minute if per_minute else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval:
            pause = self._last + self.min_interval - time.monotonic()
            if pause > 0:
                time.sleep(pause)
        self._last = time.monotonic()


def default_is_retryable(exc: Exception) -> bool:
    """HTTP 429 (too many requests) and 5xx (server hiccups) are worth retrying."""
    return getattr(exc, "code", None) in (429, 500, 502, 503, 504)


def call_with_retries(
    fn: Callable[[], T],
    *,
    max_retries: int = 5,
    base_delay: float = 2.0,
    is_retryable: Callable[[Exception], bool] = default_is_retryable,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn()`; on a retryable error wait 2s, 4s, 8s ... (plus jitter) and try again."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise anything non-retryable
            if attempt == max_retries or not is_retryable(exc):
                raise
            sleep(base_delay * (2**attempt) + random.random())
    raise AssertionError("unreachable")  # pragma: no cover


def to_jsonable(obj: Any) -> Any:
    """Convert numpy types / NaN into plain JSON values (NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
