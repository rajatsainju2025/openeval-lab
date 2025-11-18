"""Memory profiling and leak detection.

Uses tracemalloc for detailed memory analysis.
"""

import tracemalloc
from typing import List, Tuple, Optional


class MemoryProfiler:
    """Track memory usage and detect leaks."""

    def __init__(self, enabled: bool = True):
        """Initialize memory profiler."""
        self.enabled = enabled
        self.snapshots: List[tracemalloc.Snapshot] = []
        if enabled and not tracemalloc.is_tracing():
            tracemalloc.start()

    def take_snapshot(self, label: str = "") -> Optional[tracemalloc.Snapshot]:
        """Take a memory snapshot."""
        if not self.enabled:
            return None

        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append(snapshot)
        return snapshot

    def get_top_allocations(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get top memory allocations."""
        if not self.snapshots:
            return []

        snapshot = self.snapshots[-1]
        stats = snapshot.statistics("lineno")

        return [(str(stat), stat.size) for stat in stats[:limit]]

    def get_memory_growth(self) -> Optional[int]:
        """Calculate memory growth since first snapshot."""
        if len(self.snapshots) < 2:
            return None

        first = self.snapshots[0]
        last = self.snapshots[-1]

        first_stats = {s.traceback: s.size for s in first.statistics("lineno")}
        last_stats = {s.traceback: s.size for s in last.statistics("lineno")}

        growth = sum(last_stats.get(k, 0) - v for k, v in first_stats.items())
        return growth

    def stop(self):
        """Stop memory profiling."""
        if self.enabled:
            tracemalloc.stop()


__all__ = ["MemoryProfiler"]
