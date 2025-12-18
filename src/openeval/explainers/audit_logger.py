"""Audit logger for tracking explanation generation activities.

This module provides comprehensive audit logging for all explanation
generation activities, supporting compliance and debugging.

Example:
    >>> from openeval.explainers import AuditLogger, log_event
    >>> logger = AuditLogger()
    >>> logger.log_event("explanation_generated", {"code_id": "abc123"})
    >>> events = logger.query(event_type="explanation_generated")
"""

from __future__ import annotations

import hashlib
import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator


class AuditEventType(Enum):
    """Types of audit events."""

    EXPLANATION_REQUESTED = "explanation_requested"
    EXPLANATION_GENERATED = "explanation_generated"
    EXPLANATION_CACHED = "explanation_cached"
    EXPLANATION_RETRIEVED = "explanation_retrieved"
    EXPLANATION_UPDATED = "explanation_updated"
    EXPLANATION_DELETED = "explanation_deleted"
    CODE_ANALYZED = "code_analyzed"
    ERROR_OCCURRED = "error_occurred"
    CONFIG_CHANGED = "config_changed"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"


class AuditSeverity(Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditAction(Enum):
    """Actions that can be audited."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    CONFIGURE = "configure"
    AUTHENTICATE = "authenticate"
    AUTHORIZE = "authorize"


@dataclass
class AuditEvent:
    """An audit log event."""

    event_id: str
    event_type: AuditEventType
    action: AuditAction
    severity: AuditSeverity
    timestamp: datetime
    user_id: str | None = None
    session_id: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_ip: str | None = None
    user_agent: str | None = None
    duration_ms: int | None = None
    success: bool = True
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "action": self.action.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "metadata": self.metadata,
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEvent:
        """Create event from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=AuditEventType(data["event_type"]),
            action=AuditAction(data["action"]),
            severity=AuditSeverity(data["severity"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            resource_type=data.get("resource_type"),
            resource_id=data.get("resource_id"),
            details=data.get("details", {}),
            metadata=data.get("metadata", {}),
            source_ip=data.get("source_ip"),
            user_agent=data.get("user_agent"),
            duration_ms=data.get("duration_ms"),
            success=data.get("success", True),
            error_message=data.get("error_message"),
        )


@dataclass
class AuditQuery:
    """Query parameters for audit log search."""

    event_types: list[AuditEventType] | None = None
    actions: list[AuditAction] | None = None
    severities: list[AuditSeverity] | None = None
    user_id: str | None = None
    session_id: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    success_only: bool | None = None
    limit: int = 100
    offset: int = 0


@dataclass
class AuditStats:
    """Statistics about audit events."""

    total_events: int
    events_by_type: dict[str, int]
    events_by_severity: dict[str, int]
    events_by_action: dict[str, int]
    success_rate: float
    time_range: tuple[datetime, datetime] | None


class AuditStorage(ABC):
    """Abstract base class for audit storage backends."""

    @abstractmethod
    def store(self, event: AuditEvent) -> None:
        """Store an audit event."""
        pass

    @abstractmethod
    def query(self, query: AuditQuery) -> list[AuditEvent]:
        """Query audit events."""
        pass

    @abstractmethod
    def get_by_id(self, event_id: str) -> AuditEvent | None:
        """Get an event by ID."""
        pass

    @abstractmethod
    def count(self, query: AuditQuery) -> int:
        """Count matching events."""
        pass


class InMemoryAuditStorage(AuditStorage):
    """In-memory audit storage implementation."""

    def __init__(self, max_events: int = 10000) -> None:
        """Initialize storage with max capacity."""
        self.max_events = max_events
        self._events: list[AuditEvent] = []
        self._index: dict[str, AuditEvent] = {}
        self._lock = threading.Lock()

    def store(self, event: AuditEvent) -> None:
        """Store an audit event."""
        with self._lock:
            if len(self._events) >= self.max_events:
                # Remove oldest event
                old_event = self._events.pop(0)
                del self._index[old_event.event_id]

            self._events.append(event)
            self._index[event.event_id] = event

    def query(self, query: AuditQuery) -> list[AuditEvent]:
        """Query audit events."""
        with self._lock:
            results = self._filter_events(query)
            return results[query.offset : query.offset + query.limit]

    def get_by_id(self, event_id: str) -> AuditEvent | None:
        """Get an event by ID."""
        return self._index.get(event_id)

    def count(self, query: AuditQuery) -> int:
        """Count matching events."""
        with self._lock:
            return len(self._filter_events(query))

    def _filter_events(self, query: AuditQuery) -> list[AuditEvent]:
        """Filter events by query criteria."""
        results = []

        for event in self._events:
            if query.event_types and event.event_type not in query.event_types:
                continue
            if query.actions and event.action not in query.actions:
                continue
            if query.severities and event.severity not in query.severities:
                continue
            if query.user_id and event.user_id != query.user_id:
                continue
            if query.session_id and event.session_id != query.session_id:
                continue
            if query.resource_type and event.resource_type != query.resource_type:
                continue
            if query.resource_id and event.resource_id != query.resource_id:
                continue
            if query.start_time and event.timestamp < query.start_time:
                continue
            if query.end_time and event.timestamp > query.end_time:
                continue
            if query.success_only is not None and event.success != query.success_only:
                continue

            results.append(event)

        return results


class FileAuditStorage(AuditStorage):
    """File-based audit storage implementation."""

    def __init__(self, base_path: Path | str, rotate_daily: bool = True) -> None:
        """Initialize file storage."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.rotate_daily = rotate_daily
        self._lock = threading.Lock()

    def _get_log_file(self, timestamp: datetime | None = None) -> Path:
        """Get the log file path for a timestamp."""
        if timestamp is None:
            timestamp = datetime.now()

        if self.rotate_daily:
            filename = f"audit_{timestamp.strftime('%Y%m%d')}.jsonl"
        else:
            filename = "audit.jsonl"

        return self.base_path / filename

    def store(self, event: AuditEvent) -> None:
        """Store an audit event."""
        log_file = self._get_log_file(event.timestamp)

        with self._lock:
            with open(log_file, "a") as file_handle:
                json_line = json.dumps(event.to_dict())
                file_handle.write(json_line + "\n")

    def query(self, query: AuditQuery) -> list[AuditEvent]:
        """Query audit events."""
        results: list[AuditEvent] = []

        # Determine which files to search
        files_to_search = self._get_files_for_query(query)

        for log_file in files_to_search:
            if not log_file.exists():
                continue

            with open(log_file) as file_handle:
                for line in file_handle:
                    try:
                        data = json.loads(line.strip())
                        event = AuditEvent.from_dict(data)

                        if self._matches_query(event, query):
                            results.append(event)
                    except (json.JSONDecodeError, KeyError):
                        continue

        # Sort by timestamp descending
        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results[query.offset : query.offset + query.limit]

    def get_by_id(self, event_id: str) -> AuditEvent | None:
        """Get an event by ID."""
        for log_file in self.base_path.glob("audit_*.jsonl"):
            with open(log_file) as file_handle:
                for line in file_handle:
                    try:
                        data = json.loads(line.strip())
                        if data.get("event_id") == event_id:
                            return AuditEvent.from_dict(data)
                    except (json.JSONDecodeError, KeyError):
                        continue
        return None

    def count(self, query: AuditQuery) -> int:
        """Count matching events."""
        count = 0
        files_to_search = self._get_files_for_query(query)

        for log_file in files_to_search:
            if not log_file.exists():
                continue

            with open(log_file) as file_handle:
                for line in file_handle:
                    try:
                        data = json.loads(line.strip())
                        event = AuditEvent.from_dict(data)
                        if self._matches_query(event, query):
                            count += 1
                    except (json.JSONDecodeError, KeyError):
                        continue

        return count

    def _get_files_for_query(self, query: AuditQuery) -> list[Path]:
        """Get log files relevant to query time range."""
        if not self.rotate_daily:
            return [self.base_path / "audit.jsonl"]

        files = []
        if query.start_time and query.end_time:
            current = query.start_time
            while current <= query.end_time:
                files.append(self._get_log_file(current))
                current += timedelta(days=1)
        else:
            files = list(self.base_path.glob("audit_*.jsonl"))

        return files

    def _matches_query(self, event: AuditEvent, query: AuditQuery) -> bool:
        """Check if event matches query criteria."""
        if query.event_types and event.event_type not in query.event_types:
            return False
        if query.actions and event.action not in query.actions:
            return False
        if query.severities and event.severity not in query.severities:
            return False
        if query.user_id and event.user_id != query.user_id:
            return False
        if query.session_id and event.session_id != query.session_id:
            return False
        if query.resource_type and event.resource_type != query.resource_type:
            return False
        if query.resource_id and event.resource_id != query.resource_id:
            return False
        if query.start_time and event.timestamp < query.start_time:
            return False
        if query.end_time and event.timestamp > query.end_time:
            return False
        if query.success_only is not None and event.success != query.success_only:
            return False
        return True


class AuditLogger:
    """Main audit logger class."""

    def __init__(
        self,
        storage: AuditStorage | None = None,
        default_user_id: str | None = None,
    ) -> None:
        """Initialize the audit logger."""
        self.storage = storage or InMemoryAuditStorage()
        self.default_user_id = default_user_id
        self._event_count = 0
        self._listeners: list[Callable[[AuditEvent], None]] = []
        self._filters: list[Callable[[AuditEvent], bool]] = []

    def log_event(
        self,
        event_type: AuditEventType | str,
        action: AuditAction | str = AuditAction.EXECUTE,
        severity: AuditSeverity | str = AuditSeverity.INFO,
        user_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
        success: bool = True,
        error_message: str | None = None,
        **metadata: Any,
    ) -> AuditEvent:
        """Log an audit event.

        Args:
            event_type: Type of event.
            action: Action performed.
            severity: Event severity.
            user_id: User who triggered event.
            resource_type: Type of resource affected.
            resource_id: ID of resource affected.
            details: Event details.
            success: Whether action succeeded.
            error_message: Error message if failed.
            **metadata: Additional metadata.

        Returns:
            The created AuditEvent.
        """
        # Convert string types to enums
        if isinstance(event_type, str):
            event_type = AuditEventType(event_type)
        if isinstance(action, str):
            action = AuditAction(action)
        if isinstance(severity, str):
            severity = AuditSeverity(severity)

        self._event_count += 1
        event_id = self._generate_event_id()

        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            action=action,
            severity=severity,
            timestamp=datetime.now(),
            user_id=user_id or self.default_user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            metadata=metadata,
            success=success,
            error_message=error_message,
        )

        # Apply filters
        for filter_func in self._filters:
            if not filter_func(event):
                return event

        # Store event
        self.storage.store(event)

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass

        return event

    def log_explanation_requested(
        self,
        code_id: str,
        user_id: str | None = None,
        **details: Any,
    ) -> AuditEvent:
        """Log an explanation request."""
        return self.log_event(
            event_type=AuditEventType.EXPLANATION_REQUESTED,
            action=AuditAction.CREATE,
            resource_type="explanation",
            resource_id=code_id,
            user_id=user_id,
            details=details,
        )

    def log_explanation_generated(
        self,
        code_id: str,
        explanation_id: str,
        duration_ms: int,
        user_id: str | None = None,
        **details: Any,
    ) -> AuditEvent:
        """Log an explanation generation."""
        return self.log_event(
            event_type=AuditEventType.EXPLANATION_GENERATED,
            action=AuditAction.CREATE,
            resource_type="explanation",
            resource_id=explanation_id,
            user_id=user_id,
            duration_ms=duration_ms,
            details={"code_id": code_id, **details},
        )

    def log_error(
        self,
        error_message: str,
        resource_type: str | None = None,
        resource_id: str | None = None,
        **details: Any,
    ) -> AuditEvent:
        """Log an error event."""
        return self.log_event(
            event_type=AuditEventType.ERROR_OCCURRED,
            action=AuditAction.EXECUTE,
            severity=AuditSeverity.ERROR,
            resource_type=resource_type,
            resource_id=resource_id,
            success=False,
            error_message=error_message,
            details=details,
        )

    def log_security_event(
        self,
        action: str,
        user_id: str | None = None,
        success: bool = True,
        **details: Any,
    ) -> AuditEvent:
        """Log a security event."""
        return self.log_event(
            event_type=AuditEventType.SECURITY_EVENT,
            action=AuditAction.AUTHENTICATE if "auth" in action.lower() else AuditAction.AUTHORIZE,
            severity=AuditSeverity.WARNING if not success else AuditSeverity.INFO,
            user_id=user_id,
            success=success,
            details={"security_action": action, **details},
        )

    def query(
        self,
        event_type: AuditEventType | None = None,
        action: AuditAction | None = None,
        severity: AuditSeverity | None = None,
        user_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> list[AuditEvent]:
        """Query audit events.

        Args:
            event_type: Filter by event type.
            action: Filter by action.
            severity: Filter by severity.
            user_id: Filter by user.
            start_time: Filter by start time.
            end_time: Filter by end time.
            limit: Maximum results.
            **kwargs: Additional filters.

        Returns:
            List of matching events.
        """
        query = AuditQuery(
            event_types=[event_type] if event_type else None,
            actions=[action] if action else None,
            severities=[severity] if severity else None,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            **kwargs,
        )
        return self.storage.query(query)

    def get_event(self, event_id: str) -> AuditEvent | None:
        """Get an event by ID."""
        return self.storage.get_by_id(event_id)

    def add_listener(self, listener: Callable[[AuditEvent], None]) -> None:
        """Add an event listener."""
        self._listeners.append(listener)

    def add_filter(self, filter_func: Callable[[AuditEvent], bool]) -> None:
        """Add an event filter."""
        self._filters.append(filter_func)

    def get_stats(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> AuditStats:
        """Get audit statistics."""
        query = AuditQuery(start_time=start_time, end_time=end_time, limit=10000)
        events = self.storage.query(query)

        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_action: dict[str, int] = {}
        success_count = 0
        timestamps: list[datetime] = []

        for event in events:
            by_type[event.event_type.value] = by_type.get(event.event_type.value, 0) + 1
            by_severity[event.severity.value] = by_severity.get(event.severity.value, 0) + 1
            by_action[event.action.value] = by_action.get(event.action.value, 0) + 1
            if event.success:
                success_count += 1
            timestamps.append(event.timestamp)

        time_range = None
        if timestamps:
            time_range = (min(timestamps), max(timestamps))

        return AuditStats(
            total_events=len(events),
            events_by_type=by_type,
            events_by_severity=by_severity,
            events_by_action=by_action,
            success_rate=success_count / len(events) if events else 1.0,
            time_range=time_range,
        )

    def iterate_events(self, query: AuditQuery | None = None) -> Iterator[AuditEvent]:
        """Iterate over audit events."""
        query = query or AuditQuery()
        events = self.storage.query(query)
        yield from events

    def _generate_event_id(self) -> str:
        """Generate a unique event ID."""
        content = f"{self._event_count}:{datetime.now().isoformat()}"
        return f"evt_{hashlib.md5(content.encode()).hexdigest()[:12]}"


# Global instance
_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def reset_audit_logger() -> None:
    """Reset the global audit logger."""
    global _audit_logger
    _audit_logger = None


def create_audit_logger(
    storage: AuditStorage | None = None,
    default_user_id: str | None = None,
) -> AuditLogger:
    """Create a new audit logger."""
    return AuditLogger(storage=storage, default_user_id=default_user_id)


def log_event(
    event_type: AuditEventType | str,
    action: AuditAction | str = AuditAction.EXECUTE,
    **kwargs: Any,
) -> AuditEvent:
    """Log an audit event using global logger."""
    return get_audit_logger().log_event(event_type=event_type, action=action, **kwargs)


def create_file_audit_storage(path: str | Path) -> FileAuditStorage:
    """Create a file-based audit storage."""
    return FileAuditStorage(path)


def create_audit_query(
    event_types: list[AuditEventType] | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    **kwargs: Any,
) -> AuditQuery:
    """Create an audit query."""
    return AuditQuery(
        event_types=event_types,
        start_time=start_time,
        end_time=end_time,
        **kwargs,
    )
