"""Async batch processing for benchmarks.

Replaces synchronous loops with asyncio for parallel adapter evaluation.
"""

import asyncio
from typing import List, Dict, Any


async def run_adapters_async(adapters: List[Any]) -> List[Dict[str, Any]]:
    """Run multiple adapters concurrently using asyncio.gather()."""
    tasks = [adapter.run_async() for adapter in adapters]
    return await asyncio.gather(*tasks)


def run_benchmark_batch(adapters: List[Any]) -> List[Dict[str, Any]]:
    """Run benchmark batch with async execution."""
    loop = asyncio.new_event_loop()
    try:
        results = loop.run_until_complete(run_adapters_async(adapters))
        return results
    finally:
        loop.close()


__all__ = ["run_adapters_async", "run_benchmark_batch"]
