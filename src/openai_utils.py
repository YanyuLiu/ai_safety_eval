"""OpenAI API helpers with rate-limit handling."""

from __future__ import annotations

import time
from typing import Callable, TypeVar

from openai import APIStatusError, OpenAI, RateLimitError

T = TypeVar("T")

_last_call_time = 0.0
MIN_INTERVAL_SEC = 6.5  # stay under 10 RPM tier
MAX_RETRIES = 8


def call_with_retry(create_fn: Callable[[], T]) -> T:
    """Execute an OpenAI API call with pacing and retry on rate limits."""
    global _last_call_time

    for attempt in range(MAX_RETRIES):
        elapsed = time.time() - _last_call_time
        if elapsed < MIN_INTERVAL_SEC:
            time.sleep(MIN_INTERVAL_SEC - elapsed)

        try:
            _last_call_time = time.time()
            return create_fn()
        except RateLimitError:
            wait = MIN_INTERVAL_SEC * (attempt + 1)
            print(f"  Rate limited — waiting {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
        except APIStatusError as e:
            if e.status_code == 429:
                wait = MIN_INTERVAL_SEC * (attempt + 1)
                print(f"  Rate limited — waiting {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("OpenAI API rate limit exceeded after retries")
