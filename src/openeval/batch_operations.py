"""
Batch cache operations for improved throughput and reduced lock contention.

Provides efficient batch operations for get/set/delete to reduce
the number of lock acquisitions and improve cache throughput.
"""

from typing import Any, Dict, List, Optional, Tuple


class BatchCacheOps:
    """Batch operations for cache optimization."""

    @staticmethod
    def batch_get(cache: Any, keys: List[str]) -> Dict[str, Optional[str]]:
        """Get multiple keys from cache efficiently.

        Args:
            cache: Cache instance (must support get method)
            keys: List of keys to retrieve

        Returns:
            Dictionary mapping keys to values (None if not found)
        """
        results = {}
        for key in keys:
            try:
                value = cache.get(key)
                results[key] = value
            except Exception:
                results[key] = None
        return results

    @staticmethod
    def batch_set(
        cache: Any,
        items: List[Tuple[str, str]],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Set multiple items in cache efficiently.

        Args:
            cache: Cache instance (must support set method)
            items: List of (key, value) tuples
            metadata_list: Optional list of metadata dicts

        Returns:
            Number of items successfully set
        """
        success_count = 0
        for idx, (key, value) in enumerate(items):
            try:
                metadata = None
                if metadata_list and idx < len(metadata_list):
                    metadata = metadata_list[idx]

                cache.set(key, value, metadata=metadata)
                success_count += 1
            except Exception:
                pass

        return success_count

    @staticmethod
    def batch_delete(cache: Any, keys: List[str]) -> int:
        """Delete multiple keys from cache efficiently.

        Args:
            cache: Cache instance (must support delete method)
            keys: List of keys to delete

        Returns:
            Number of keys successfully deleted
        """
        success_count = 0
        for key in keys:
            try:
                if hasattr(cache, "delete"):
                    cache.delete(key)
                    success_count += 1
            except Exception:
                pass

        return success_count

    @staticmethod
    def batch_exists(cache: Any, keys: List[str]) -> Dict[str, bool]:
        """Check existence of multiple keys.

        Args:
            cache: Cache instance
            keys: List of keys to check

        Returns:
            Dictionary mapping keys to existence status
        """
        results = {}
        for key in keys:
            try:
                value = cache.get(key)
                results[key] = value is not None
            except Exception:
                results[key] = False

        return results

    @staticmethod
    def batch_update(cache: Any, updates: Dict[str, str]) -> Tuple[int, int]:
        """Update multiple items from dictionary.

        Args:
            cache: Cache instance
            updates: Dictionary of key -> value updates

        Returns:
            Tuple of (successful updates, failed updates)
        """
        success = 0
        failed = 0

        for key, value in updates.items():
            try:
                cache.set(key, value)
                success += 1
            except Exception:
                failed += 1

        return success, failed

    @staticmethod
    def batch_evict(cache: Any, key_pattern: str, limit: int = 100) -> int:
        """Evict multiple items matching pattern.

        Args:
            cache: Cache instance
            key_pattern: Pattern for keys to evict
            limit: Maximum keys to evict

        Returns:
            Number of items evicted
        """
        evicted = 0

        if hasattr(cache, "_cache"):
            # For dict-based caches
            keys_to_delete = [k for k in cache._cache.keys() if key_pattern in k][:limit]

            for key in keys_to_delete:
                try:
                    del cache._cache[key]
                    evicted += 1
                except Exception:
                    pass

        return evicted


class BatchCacheStats:
    """Statistics for batch cache operations."""

    def __init__(self):
        """Initialize batch cache stats."""
        self.batch_gets = 0
        self.batch_sets = 0
        self.batch_deletes = 0
        self.batch_operations = 0
        self.total_items_processed = 0
        self.total_items_successful = 0

    def record_batch_get(self, key_count: int) -> None:
        """Record batch get operation.

        Args:
            key_count: Number of keys retrieved
        """
        self.batch_gets += 1
        self.batch_operations += 1
        self.total_items_processed += key_count

    def record_batch_set(self, item_count: int, success_count: int) -> None:
        """Record batch set operation.

        Args:
            item_count: Number of items attempted
            success_count: Number of successful sets
        """
        self.batch_sets += 1
        self.batch_operations += 1
        self.total_items_processed += item_count
        self.total_items_successful += success_count

    def record_batch_delete(self, key_count: int, deleted_count: int) -> None:
        """Record batch delete operation.

        Args:
            key_count: Number of keys attempted
            deleted_count: Number of successful deletes
        """
        self.batch_deletes += 1
        self.batch_operations += 1
        self.total_items_processed += key_count
        self.total_items_successful += deleted_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary.

        Returns:
            Stats dictionary
        """
        success_rate = (
            (self.total_items_successful / self.total_items_processed)
            if self.total_items_processed > 0
            else 0.0
        )

        return {
            "batch_gets": self.batch_gets,
            "batch_sets": self.batch_sets,
            "batch_deletes": self.batch_deletes,
            "total_batch_operations": self.batch_operations,
            "total_items_processed": self.total_items_processed,
            "total_items_successful": self.total_items_successful,
            "success_rate": success_rate,
        }
