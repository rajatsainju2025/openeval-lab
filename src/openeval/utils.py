from __future__ import annotations

import os
import random
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FTimeoutError
from pathlib import Path
from typing import Callable, Optional, TypeVar


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass


def hash_file(path: Path | str, *, algo: str = "sha256", chunk_size: int = 1 << 20) -> str:
    p = Path(path)
    h = hashlib.new(algo)
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hash_prompt(key_parts: list[str], algo: str = "sha256") -> str:
    """Hash adapter/model/prompt/kwargs into a stable cache key.

    key_parts: ordered list of stable strings (e.g., adapter name, model, prompt, sorted kwargs JSON)
    """
    h = hashlib.new(algo)
    for part in key_parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x1f")  # unit separator
    return h.hexdigest()


T = TypeVar("T")


def retry_call(
    fn: Callable[[], T],
    *,
    retries: int = 0,
    base_delay: float = 0.2,
    max_delay: float = 5.0,
    jitter: float = 0.1,
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
) -> T:
    """Call fn with simple exponential backoff and jitter.

    retries: number of retries after the first attempt (total attempts = retries+1)
    base_delay: initial delay in seconds
    max_delay: max backoff delay
    jitter: random jitter factor in seconds added to delay
    on_retry: callback with (attempt_index, exception)
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:  # pragma: no cover - difficult to fully branch
            if attempt >= retries:
                raise
            if on_retry:
                try:
                    on_retry(attempt + 1, e)
                except Exception:
                    pass
            delay = min(max_delay, base_delay * (2**attempt)) + random.uniform(0, jitter)
            time.sleep(delay)
            attempt += 1


def run_with_timeout(fn: Callable[[], T], timeout: Optional[float]) -> T:
    """Run a synchronous callable with a timeout using a single-use ThreadPoolExecutor."""
    if timeout is None or timeout <= 0:
        return fn()
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout)
        except _FTimeoutError as e:  # pragma: no cover - timing sensitive
            fut.cancel()
            raise TimeoutError(f"Operation timed out after {timeout} seconds") from e
