from __future__ import annotations

import os
import random
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FTimeoutError, as_completed
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Tuple, TypeVar


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


def get_project_root(env_var: str = "OPENEVAL_PROJECT_ROOT") -> Path:
    """Return the project repository root.

    Priority:
    1) Environment variable OPENEVAL_PROJECT_ROOT if set and exists.
    2) Nearest ancestor containing pyproject.toml or .git from this file.
    3) Current working directory.
    """
    # 1) Environment override
    p_env = os.getenv(env_var)
    if p_env:
        p = Path(p_env)
        if p.exists():
            return p.resolve()

    # 2) Search upwards from this file
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent

    # 3) Fallback to CWD
    return Path.cwd()


# =============================================================================
# Batch Processing Utilities
# =============================================================================

R = TypeVar("R")


def batch_items(items: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    """Split items into batches of specified size.

    Args:
        items: Iterable of items to batch.
        batch_size: Maximum number of items per batch.

    Yields:
        Lists of items, each with at most batch_size elements.

    Example:
        >>> list(batch_items([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
    """
    batch: List[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def parallel_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int = 4,
    timeout: Optional[float] = None,
    on_error: Optional[Callable[[T, Exception], Optional[R]]] = None,
) -> List[Tuple[T, Optional[R], Optional[Exception]]]:
    """Apply function to items in parallel with error handling.

    Args:
        fn: Function to apply to each item.
        items: Iterable of items to process.
        max_workers: Maximum number of concurrent workers.
        timeout: Overall timeout for all operations (None for no timeout).
        on_error: Optional callback for handling errors. If returns a value,
                  it will be used as the result. If None, the error is recorded.

    Returns:
        List of tuples (item, result, error) for each item.
        - On success: (item, result, None)
        - On error: (item, None, exception) or (item, fallback_result, None) if on_error returns

    Example:
        >>> def process(x): return x * 2
        >>> results = parallel_map(process, [1, 2, 3], max_workers=2)
        >>> [(item, result) for item, result, _ in results]
        [(1, 2), (2, 4), (3, 6)]
    """
    items_list = list(items)
    results: List[Tuple[T, Optional[R], Optional[Exception]]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {executor.submit(fn, item): item for item in items_list}

        # Collect results
        for future in as_completed(future_to_item, timeout=timeout):
            item = future_to_item[future]
            try:
                result = future.result()
                results.append((item, result, None))
            except Exception as e:
                if on_error:
                    try:
                        fallback = on_error(item, e)
                        results.append((item, fallback, None))
                    except Exception as e2:
                        results.append((item, None, e2))
                else:
                    results.append((item, None, e))

    return results


def timed_operation(operation_name: str = "operation"):
    """Decorator to time a function and log duration.

    Args:
        operation_name: Name of the operation for logging.

    Returns:
        Decorator that times the wrapped function.

    Example:
        >>> @timed_operation("data_loading")
        ... def load_data():
        ...     return [1, 2, 3]
    """

    def decorator(fn: Callable[..., R]) -> Callable[..., R]:
        def wrapper(*args, **kwargs) -> R:
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                # Use print for simplicity; could integrate with logging
                if elapsed > 1.0:
                    print(f"[TIMING] {operation_name}: {elapsed:.2f}s")

        return wrapper

    return decorator


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default on division by zero.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value to return if denominator is zero.

    Returns:
        The result of division or default.

    Example:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        0.0
    """
    if denominator == 0:
        return default
    return numerator / denominator


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human-readable duration string.

    Example:
        >>> format_duration(3661.5)
        '1h 1m 1.50s'
        >>> format_duration(45.3)
        '45.30s'
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"
