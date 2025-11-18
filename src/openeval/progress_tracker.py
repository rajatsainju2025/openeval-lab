"""Progress tracking with ETA estimation.

Calculates and displays progress with time remaining.
"""

import time
from typing import Optional


class ProgressTracker:
    """Track evaluation progress with ETA."""

    def __init__(self, total_items: int):
        """Initialize progress tracker."""
        self.total_items = total_items
        self.completed_items = 0
        self.start_time = time.time()
        self.item_times = []

    def update(self, items_completed: int = 1):
        """Update progress."""
        now = time.time()
        self.completed_items += items_completed

        # Track time per item
        if self.completed_items > 0:
            elapsed = now - self.start_time
            avg_time = elapsed / self.completed_items
            self.item_times.append(avg_time)

    def get_eta_seconds(self) -> Optional[float]:
        """Calculate estimated time to completion."""
        if not self.item_times or self.completed_items == 0:
            return None

        remaining = self.total_items - self.completed_items
        if remaining <= 0:
            return 0.0

        avg_time = sum(self.item_times[-10:]) / len(self.item_times[-10:])
        return remaining * avg_time

    def get_eta_string(self) -> str:
        """Get human-readable ETA."""
        eta = self.get_eta_seconds()
        if eta is None:
            return "unknown"

        if eta < 60:
            return f"{int(eta)}s"
        elif eta < 3600:
            return f"{int(eta / 60)}m {int(eta % 60)}s"
        else:
            hours = int(eta / 3600)
            minutes = int((eta % 3600) / 60)
            return f"{hours}h {minutes}m"

    def get_progress_percent(self) -> float:
        """Get completion percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100

    def get_status(self) -> str:
        """Get status string."""
        percent = self.get_progress_percent()
        eta = self.get_eta_string()
        return f"{percent:.1f}% ({self.completed_items}/{self.total_items}) ETA: {eta}"


__all__ = ["ProgressTracker"]
