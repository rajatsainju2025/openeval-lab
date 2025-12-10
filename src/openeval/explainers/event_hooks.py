"""Event system for explainer lifecycle hooks.

Enables subscribing to and emitting events during explanation generation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ExplainerEventType(str, Enum):
    """Types of events emitted during explanation lifecycle."""

    # Request events
    EXPLAIN_START = "explain_start"
    EXPLAIN_END = "explain_end"
    EXPLAIN_ERROR = "explain_error"

    # Cache events
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CACHE_SET = "cache_set"

    # Chain events
    CHAIN_START = "chain_start"
    CHAIN_EXPLAINER_START = "chain_explainer_start"
    CHAIN_EXPLAINER_END = "chain_explainer_end"
    CHAIN_FALLBACK = "chain_fallback"
    CHAIN_END = "chain_end"

    # Quality events
    QUALITY_CHECK_START = "quality_check_start"
    QUALITY_CHECK_END = "quality_check_end"

    # Model events
    MODEL_SELECTED = "model_selected"
    TOKEN_USAGE = "token_usage"


@dataclass
class ExplainerEvent:
    """An event emitted during explanation lifecycle."""

    event_type: ExplainerEventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "source": self.source,
            "correlation_id": self.correlation_id,
        }


# Type alias for event handlers
EventHandler = Callable[[ExplainerEvent], None]


class EventEmitter:
    """Event emitter for explainer lifecycle events.

    Allows subscribing to specific event types and emitting events.
    Supports both sync handlers and wildcard subscriptions.
    """

    def __init__(self) -> None:
        """Initialize event emitter."""
        self._handlers: Dict[ExplainerEventType, List[EventHandler]] = {}
        self._wildcard_handlers: List[EventHandler] = []
        self._event_history: List[ExplainerEvent] = []
        self._max_history: int = 1000
        self._enabled: bool = True

    def on(
        self,
        event_type: ExplainerEventType,
        handler: EventHandler,
    ) -> "EventEmitter":
        """Subscribe to a specific event type.

        Args:
            event_type: Type of event to subscribe to.
            handler: Callback function to invoke when event is emitted.

        Returns:
            Self for method chaining.
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        return self

    def on_all(self, handler: EventHandler) -> "EventEmitter":
        """Subscribe to all events.

        Args:
            handler: Callback function to invoke for any event.

        Returns:
            Self for method chaining.
        """
        self._wildcard_handlers.append(handler)
        return self

    def off(
        self,
        event_type: ExplainerEventType,
        handler: Optional[EventHandler] = None,
    ) -> "EventEmitter":
        """Unsubscribe from an event type.

        Args:
            event_type: Type of event to unsubscribe from.
            handler: Specific handler to remove (all if None).

        Returns:
            Self for method chaining.
        """
        if event_type in self._handlers:
            if handler:
                self._handlers[event_type] = [h for h in self._handlers[event_type] if h != handler]
            else:
                self._handlers[event_type] = []
        return self

    def off_all(self, handler: Optional[EventHandler] = None) -> "EventEmitter":
        """Unsubscribe from all events.

        Args:
            handler: Specific handler to remove (all if None).

        Returns:
            Self for method chaining.
        """
        if handler:
            self._wildcard_handlers = [h for h in self._wildcard_handlers if h != handler]
        else:
            self._wildcard_handlers = []
        return self

    def emit(self, event: ExplainerEvent) -> None:
        """Emit an event to all subscribers.

        Args:
            event: Event to emit.
        """
        if not self._enabled:
            return

        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history :]

        # Notify specific handlers
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                # Don't let handler errors break the flow
                pass

        # Notify wildcard handlers
        for handler in self._wildcard_handlers:
            try:
                handler(event)
            except Exception:
                pass

    def emit_simple(
        self,
        event_type: ExplainerEventType,
        data: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Emit an event with simplified parameters.

        Args:
            event_type: Type of event.
            data: Event data dictionary.
            source: Source of the event.
            correlation_id: Correlation ID for tracing.
        """
        event = ExplainerEvent(
            event_type=event_type,
            data=data or {},
            source=source,
            correlation_id=correlation_id,
        )
        self.emit(event)

    def enable(self) -> None:
        """Enable event emission."""
        self._enabled = True

    def disable(self) -> None:
        """Disable event emission."""
        self._enabled = False

    def get_history(
        self,
        event_type: Optional[ExplainerEventType] = None,
        limit: Optional[int] = None,
    ) -> List[ExplainerEvent]:
        """Get event history.

        Args:
            event_type: Filter by event type (all if None).
            limit: Maximum number of events to return.

        Returns:
            List of events (newest last).
        """
        events = self._event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if limit:
            events = events[-limit:]
        return events

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history = []

    def get_stats(self) -> Dict[str, Any]:
        """Get event statistics.

        Returns:
            Dictionary with event stats.
        """
        type_counts: Dict[str, int] = {}
        for event in self._event_history:
            key = event.event_type.value
            type_counts[key] = type_counts.get(key, 0) + 1

        return {
            "total_events": len(self._event_history),
            "events_by_type": type_counts,
            "enabled": self._enabled,
            "registered_handlers": sum(len(h) for h in self._handlers.values()),
            "wildcard_handlers": len(self._wildcard_handlers),
        }


class EventSubscriber:
    """Decorator-based event subscription helper.

    Allows using decorators to subscribe to events.
    """

    def __init__(self, emitter: EventEmitter) -> None:
        """Initialize subscriber.

        Args:
            emitter: EventEmitter to subscribe to.
        """
        self.emitter = emitter

    def on(self, event_type: ExplainerEventType) -> Callable:
        """Decorator to subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to.

        Returns:
            Decorator function.
        """

        def decorator(func: EventHandler) -> EventHandler:
            self.emitter.on(event_type, func)
            return func

        return decorator

    def on_all(self) -> Callable:
        """Decorator to subscribe to all events.

        Returns:
            Decorator function.
        """

        def decorator(func: EventHandler) -> EventHandler:
            self.emitter.on_all(func)
            return func

        return decorator


# Global event emitter instance
_global_event_emitter = EventEmitter()


def get_event_emitter() -> EventEmitter:
    """Get the global event emitter instance.

    Returns:
        EventEmitter singleton.
    """
    return _global_event_emitter


def get_event_subscriber() -> EventSubscriber:
    """Get an event subscriber for the global emitter.

    Returns:
        EventSubscriber for global emitter.
    """
    return EventSubscriber(_global_event_emitter)
