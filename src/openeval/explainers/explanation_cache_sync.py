"""Distributed cache synchronization for code explanations.

This module provides functionality for synchronizing explanation caches
across multiple nodes in a distributed system.

Example:
    >>> from openeval.explainers import CacheSyncManager, sync_cache_entry
    >>> manager = CacheSyncManager()
    >>> await manager.sync_entry("key", "explanation data")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable


class SyncStrategy(Enum):
    """Cache synchronization strategies."""

    IMMEDIATE = "immediate"
    EVENTUAL = "eventual"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    INVALIDATE = "invalidate"


class ConflictResolution(Enum):
    """Conflict resolution strategies."""

    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    VERSION_VECTOR = "version_vector"
    MERGE = "merge"
    CUSTOM = "custom"


class NodeStatus(Enum):
    """Node status in the cluster."""

    ONLINE = "online"
    OFFLINE = "offline"
    SYNCING = "syncing"
    DEGRADED = "degraded"


@dataclass
class CacheEntry:
    """A cache entry with metadata."""

    key: str
    value: Any
    version: int
    created_at: datetime
    updated_at: datetime
    ttl: int | None = None
    checksum: str = ""
    origin_node: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate checksum after initialization."""
        if not self.checksum:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate entry checksum."""
        content = json.dumps(self.value, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        expiry = self.updated_at + timedelta(seconds=self.ttl)
        return datetime.now() > expiry


@dataclass
class SyncEvent:
    """A cache synchronization event."""

    event_id: str
    event_type: str
    entry_key: str
    entry_value: Any
    version: int
    source_node: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeInfo:
    """Information about a cache node."""

    node_id: str
    address: str
    port: int
    status: NodeStatus
    last_heartbeat: datetime
    version_vector: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncConfig:
    """Configuration for cache synchronization."""

    strategy: SyncStrategy = SyncStrategy.EVENTUAL
    conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITE_WINS
    sync_interval: float = 5.0
    heartbeat_interval: float = 10.0
    max_retry_attempts: int = 3
    retry_delay: float = 1.0
    batch_size: int = 100
    enable_compression: bool = True


@dataclass
class SyncResult:
    """Result of a synchronization operation."""

    success: bool
    entries_synced: int
    conflicts_resolved: int
    errors: list[str]
    duration: float
    timestamp: datetime = field(default_factory=datetime.now)


class CacheTransport(ABC):
    """Abstract base class for cache transport mechanisms."""

    @abstractmethod
    async def send(self, node: NodeInfo, event: SyncEvent) -> bool:
        """Send an event to a node."""
        pass

    @abstractmethod
    async def receive(self) -> SyncEvent | None:
        """Receive an event."""
        pass

    @abstractmethod
    async def broadcast(self, event: SyncEvent, nodes: list[NodeInfo]) -> dict[str, bool]:
        """Broadcast an event to multiple nodes."""
        pass


class InMemoryTransport(CacheTransport):
    """In-memory transport for testing."""

    def __init__(self) -> None:
        """Initialize the transport."""
        self._queue: asyncio.Queue[SyncEvent] = asyncio.Queue()
        self._node_queues: dict[str, asyncio.Queue[SyncEvent]] = {}

    async def send(self, node: NodeInfo, event: SyncEvent) -> bool:
        """Send an event to a node."""
        if node.node_id not in self._node_queues:
            self._node_queues[node.node_id] = asyncio.Queue()
        await self._node_queues[node.node_id].put(event)
        return True

    async def receive(self) -> SyncEvent | None:
        """Receive an event."""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    async def broadcast(self, event: SyncEvent, nodes: list[NodeInfo]) -> dict[str, bool]:
        """Broadcast an event to multiple nodes."""
        results = {}
        for node in nodes:
            results[node.node_id] = await self.send(node, event)
        return results


class ConflictResolver:
    """Resolves conflicts between cache entries."""

    def __init__(self, strategy: ConflictResolution) -> None:
        """Initialize the resolver."""
        self.strategy = strategy
        self._custom_resolver: Callable[[CacheEntry, CacheEntry], CacheEntry] | None = None

    def set_custom_resolver(self, resolver: Callable[[CacheEntry, CacheEntry], CacheEntry]) -> None:
        """Set a custom conflict resolver."""
        self._custom_resolver = resolver

    def resolve(self, local: CacheEntry, remote: CacheEntry) -> CacheEntry:
        """Resolve conflict between local and remote entries."""
        if self.strategy == ConflictResolution.LAST_WRITE_WINS:
            return remote if remote.updated_at > local.updated_at else local
        elif self.strategy == ConflictResolution.FIRST_WRITE_WINS:
            return local if local.created_at < remote.created_at else remote
        elif self.strategy == ConflictResolution.VERSION_VECTOR:
            return remote if remote.version > local.version else local
        elif self.strategy == ConflictResolution.MERGE:
            return self._merge_entries(local, remote)
        elif self.strategy == ConflictResolution.CUSTOM and self._custom_resolver:
            return self._custom_resolver(local, remote)
        return remote

    def _merge_entries(self, local: CacheEntry, remote: CacheEntry) -> CacheEntry:
        """Merge two entries."""
        # For simple values, take the newer one
        if isinstance(local.value, dict) and isinstance(remote.value, dict):
            merged_value = {**local.value, **remote.value}
        else:
            merged_value = remote.value if remote.updated_at > local.updated_at else local.value

        return CacheEntry(
            key=local.key,
            value=merged_value,
            version=max(local.version, remote.version) + 1,
            created_at=min(local.created_at, remote.created_at),
            updated_at=datetime.now(),
            ttl=local.ttl,
            origin_node=local.origin_node,
        )


class VersionVector:
    """Version vector for tracking causality."""

    def __init__(self) -> None:
        """Initialize the version vector."""
        self._vector: dict[str, int] = {}

    def increment(self, node_id: str) -> int:
        """Increment version for a node."""
        current = self._vector.get(node_id, 0)
        self._vector[node_id] = current + 1
        return self._vector[node_id]

    def get(self, node_id: str) -> int:
        """Get version for a node."""
        return self._vector.get(node_id, 0)

    def merge(self, other: VersionVector) -> None:
        """Merge with another version vector."""
        for node_id, version in other._vector.items():
            self._vector[node_id] = max(self._vector.get(node_id, 0), version)

    def dominates(self, other: VersionVector) -> bool:
        """Check if this vector dominates another."""
        for node_id, version in other._vector.items():
            if self._vector.get(node_id, 0) < version:
                return False
        return True

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return dict(self._vector)

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> VersionVector:
        """Create from dictionary."""
        vector = cls()
        vector._vector = dict(data)
        return vector


class CacheSyncManager:
    """Main class for cache synchronization."""

    def __init__(
        self,
        node_id: str | None = None,
        config: SyncConfig | None = None,
        transport: CacheTransport | None = None,
    ) -> None:
        """Initialize the sync manager."""
        self.node_id = node_id or self._generate_node_id()
        self.config = config or SyncConfig()
        self.transport = transport or InMemoryTransport()

        self._cache: dict[str, CacheEntry] = {}
        self._version_vector = VersionVector()
        self._nodes: dict[str, NodeInfo] = {}
        self._pending_syncs: list[SyncEvent] = []
        self._event_handlers: dict[str, list[Callable[[SyncEvent], None]]] = {}
        self._conflict_resolver = ConflictResolver(self.config.conflict_resolution)
        self._running = False
        self._sync_task: asyncio.Task[None] | None = None

    def _generate_node_id(self) -> str:
        """Generate a unique node ID."""
        return f"node_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

    def get(self, key: str) -> CacheEntry | None:
        """Get a cache entry."""
        entry = self._cache.get(key)
        if entry and entry.is_expired():
            del self._cache[key]
            return None
        return entry

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        sync: bool = True,
    ) -> CacheEntry:
        """Set a cache entry."""
        version = self._version_vector.increment(self.node_id)
        now = datetime.now()

        entry = CacheEntry(
            key=key,
            value=value,
            version=version,
            created_at=now,
            updated_at=now,
            ttl=ttl,
            origin_node=self.node_id,
        )

        self._cache[key] = entry

        if sync:
            await self._propagate_entry(entry)

        return entry

    async def delete(self, key: str, sync: bool = True) -> bool:
        """Delete a cache entry."""
        if key not in self._cache:
            return False

        del self._cache[key]

        if sync:
            event = SyncEvent(
                event_id=self._generate_event_id(),
                event_type="delete",
                entry_key=key,
                entry_value=None,
                version=self._version_vector.increment(self.node_id),
                source_node=self.node_id,
                timestamp=datetime.now(),
            )
            await self._broadcast_event(event)

        return True

    async def _propagate_entry(self, entry: CacheEntry) -> None:
        """Propagate an entry to other nodes."""
        event = SyncEvent(
            event_id=self._generate_event_id(),
            event_type="set",
            entry_key=entry.key,
            entry_value=entry.value,
            version=entry.version,
            source_node=self.node_id,
            timestamp=datetime.now(),
            metadata={"ttl": entry.ttl},
        )

        if self.config.strategy == SyncStrategy.IMMEDIATE:
            await self._broadcast_event(event)
        else:
            self._pending_syncs.append(event)

    async def _broadcast_event(self, event: SyncEvent) -> dict[str, bool]:
        """Broadcast an event to all nodes."""
        online_nodes = [
            node
            for node in self._nodes.values()
            if node.status == NodeStatus.ONLINE and node.node_id != self.node_id
        ]
        return await self.transport.broadcast(event, online_nodes)

    def _generate_event_id(self) -> str:
        """Generate a unique event ID."""
        return f"evt_{self.node_id}_{time.time()}_{id(self)}"

    async def handle_event(self, event: SyncEvent) -> None:
        """Handle an incoming sync event."""
        if event.source_node == self.node_id:
            return

        if event.event_type == "set":
            await self._handle_set_event(event)
        elif event.event_type == "delete":
            await self._handle_delete_event(event)

        # Trigger event handlers
        for handler in self._event_handlers.get(event.event_type, []):
            handler(event)

    async def _handle_set_event(self, event: SyncEvent) -> None:
        """Handle a set event."""
        existing = self._cache.get(event.entry_key)

        new_entry = CacheEntry(
            key=event.entry_key,
            value=event.entry_value,
            version=event.version,
            created_at=event.timestamp,
            updated_at=event.timestamp,
            ttl=event.metadata.get("ttl"),
            origin_node=event.source_node,
        )

        if existing:
            resolved = self._conflict_resolver.resolve(existing, new_entry)
            self._cache[event.entry_key] = resolved
        else:
            self._cache[event.entry_key] = new_entry

    async def _handle_delete_event(self, event: SyncEvent) -> None:
        """Handle a delete event."""
        if event.entry_key in self._cache:
            existing = self._cache[event.entry_key]
            if existing.version <= event.version:
                del self._cache[event.entry_key]

    def register_node(self, node: NodeInfo) -> None:
        """Register a node in the cluster."""
        self._nodes[node.node_id] = node

    def unregister_node(self, node_id: str) -> None:
        """Unregister a node from the cluster."""
        if node_id in self._nodes:
            del self._nodes[node_id]

    def on_event(self, event_type: str, handler: Callable[[SyncEvent], None]) -> None:
        """Register an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    async def sync_with_node(self, node_id: str) -> SyncResult:
        """Synchronize with a specific node."""
        start_time = time.time()
        errors: list[str] = []
        entries_synced = 0
        conflicts_resolved = 0

        node = self._nodes.get(node_id)
        if not node:
            return SyncResult(
                success=False,
                entries_synced=0,
                conflicts_resolved=0,
                errors=[f"Node {node_id} not found"],
                duration=time.time() - start_time,
            )

        # Send all local entries
        for entry in self._cache.values():
            event = SyncEvent(
                event_id=self._generate_event_id(),
                event_type="set",
                entry_key=entry.key,
                entry_value=entry.value,
                version=entry.version,
                source_node=self.node_id,
                timestamp=datetime.now(),
                metadata={"ttl": entry.ttl},
            )

            success = await self.transport.send(node, event)
            if success:
                entries_synced += 1
            else:
                errors.append(f"Failed to sync entry {entry.key}")

        return SyncResult(
            success=len(errors) == 0,
            entries_synced=entries_synced,
            conflicts_resolved=conflicts_resolved,
            errors=errors,
            duration=time.time() - start_time,
        )

    async def full_sync(self) -> SyncResult:
        """Perform a full sync with all nodes."""
        start_time = time.time()
        total_synced = 0
        total_conflicts = 0
        all_errors: list[str] = []

        for node_id in self._nodes:
            result = await self.sync_with_node(node_id)
            total_synced += result.entries_synced
            total_conflicts += result.conflicts_resolved
            all_errors.extend(result.errors)

        return SyncResult(
            success=len(all_errors) == 0,
            entries_synced=total_synced,
            conflicts_resolved=total_conflicts,
            errors=all_errors,
            duration=time.time() - start_time,
        )

    async def flush_pending(self) -> int:
        """Flush pending sync events."""
        count = 0
        while self._pending_syncs:
            event = self._pending_syncs.pop(0)
            await self._broadcast_event(event)
            count += 1
        return count

    async def start(self) -> None:
        """Start the sync manager."""
        if self._running:
            return

        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())

    async def stop(self) -> None:
        """Stop the sync manager."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

    async def _sync_loop(self) -> None:
        """Main sync loop."""
        while self._running:
            try:
                # Process incoming events
                event = await self.transport.receive()
                if event:
                    await self.handle_event(event)

                # Flush pending syncs for eventual consistency
                if self.config.strategy == SyncStrategy.EVENTUAL:
                    await self.flush_pending()

                await asyncio.sleep(self.config.sync_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(self.config.retry_delay)

    def get_stats(self) -> dict[str, Any]:
        """Get sync manager statistics."""
        return {
            "node_id": self.node_id,
            "cache_size": len(self._cache),
            "pending_syncs": len(self._pending_syncs),
            "registered_nodes": len(self._nodes),
            "online_nodes": sum(1 for n in self._nodes.values() if n.status == NodeStatus.ONLINE),
            "version_vector": self._version_vector.to_dict(),
        }


# Global instance
_cache_sync_manager: CacheSyncManager | None = None


def get_cache_sync_manager() -> CacheSyncManager:
    """Get the global cache sync manager."""
    global _cache_sync_manager
    if _cache_sync_manager is None:
        _cache_sync_manager = CacheSyncManager()
    return _cache_sync_manager


def reset_cache_sync_manager() -> None:
    """Reset the global cache sync manager."""
    global _cache_sync_manager
    _cache_sync_manager = None


def create_cache_sync_manager(
    node_id: str | None = None,
    config: SyncConfig | None = None,
    transport: CacheTransport | None = None,
) -> CacheSyncManager:
    """Create a new cache sync manager.

    Args:
        node_id: Unique node identifier.
        config: Sync configuration.
        transport: Transport mechanism.

    Returns:
        New CacheSyncManager instance.
    """
    return CacheSyncManager(node_id=node_id, config=config, transport=transport)


async def sync_cache_entry(key: str, value: Any, ttl: int | None = None) -> CacheEntry:
    """Synchronize a cache entry.

    Args:
        key: Cache key.
        value: Cache value.
        ttl: Time-to-live in seconds.

    Returns:
        The created cache entry.
    """
    return await get_cache_sync_manager().set(key, value, ttl)


def create_sync_config(
    strategy: SyncStrategy = SyncStrategy.EVENTUAL,
    conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITE_WINS,
    **kwargs: Any,
) -> SyncConfig:
    """Create sync configuration.

    Args:
        strategy: Synchronization strategy.
        conflict_resolution: Conflict resolution strategy.
        **kwargs: Additional options.

    Returns:
        SyncConfig instance.
    """
    return SyncConfig(strategy=strategy, conflict_resolution=conflict_resolution, **kwargs)


def create_node_info(
    node_id: str,
    address: str = "localhost",
    port: int = 8000,
    status: NodeStatus = NodeStatus.ONLINE,
) -> NodeInfo:
    """Create node information.

    Args:
        node_id: Unique node identifier.
        address: Node address.
        port: Node port.
        status: Node status.

    Returns:
        NodeInfo instance.
    """
    return NodeInfo(
        node_id=node_id,
        address=address,
        port=port,
        status=status,
        last_heartbeat=datetime.now(),
    )
