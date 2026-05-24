from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """重试策略配置：次数与固定等待间隔。"""
    max_retries: int = 3
    wait_seconds: float = 0.5
    backoff_multiplier: float = 1.0
    max_wait_seconds: float | None = None


def run_with_retry(
    action: Callable[[], T],
    policy: RetryPolicy,
    on_retry_wait: Callable[[int, int, float, Exception], None] | None = None,
) -> T:
    last_error: Exception | None = None
    total_retries = max(1, int(policy.max_retries))
    base_wait = max(0.0, float(policy.wait_seconds))
    backoff_multiplier = max(1.0, float(policy.backoff_multiplier))
    max_wait = None if policy.max_wait_seconds is None else max(0.0, float(policy.max_wait_seconds))
    for retry_i in range(total_retries):
        try:
            return action()
        except Exception as exc:
            last_error = exc
            if retry_i < total_retries - 1:
                wait_seconds = base_wait * (backoff_multiplier**retry_i)
                if max_wait is not None:
                    wait_seconds = min(wait_seconds, max_wait)
                if on_retry_wait is not None:
                    on_retry_wait(retry_i + 1, total_retries, wait_seconds, exc)
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError("retry failed without explicit exception")
