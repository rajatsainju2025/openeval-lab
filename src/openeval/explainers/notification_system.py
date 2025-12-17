"""Notification system for event-based notifications.

This module provides a flexible notification system for sending
alerts about explanation events, errors, and completions.

Example:
    >>> from openeval.explainers import NotificationManager, send_notification
    >>> manager = NotificationManager()
    >>> manager.register_handler("email", EmailHandler())
    >>> await manager.notify("Explanation completed", channel="email")
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable


class NotificationPriority(Enum):
    """Priority levels for notifications."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationStatus(Enum):
    """Status of a notification."""

    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NotificationType(Enum):
    """Types of notifications."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    ALERT = "alert"


class ChannelType(Enum):
    """Types of notification channels."""

    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    SLACK = "slack"
    CONSOLE = "console"
    FILE = "file"
    CUSTOM = "custom"


@dataclass
class Notification:
    """A notification message."""

    notification_id: str
    title: str
    message: str
    notification_type: NotificationType = NotificationType.INFO
    priority: NotificationPriority = NotificationPriority.NORMAL
    status: NotificationStatus = NotificationStatus.PENDING
    channel: str | None = None
    recipient: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: datetime | None = None
    delivered_at: datetime | None = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NotificationResult:
    """Result of sending a notification."""

    notification_id: str
    success: bool
    channel: str
    error: str | None = None
    response: Any = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SubscriptionFilter:
    """Filter for notification subscriptions."""

    notification_types: list[NotificationType] | None = None
    priorities: list[NotificationPriority] | None = None
    channels: list[str] | None = None
    tags: list[str] | None = None

    def matches(self, notification: Notification) -> bool:
        """Check if notification matches filter."""
        if self.notification_types:
            if notification.notification_type not in self.notification_types:
                return False

        if self.priorities:
            if notification.priority not in self.priorities:
                return False

        if self.channels and notification.channel:
            if notification.channel not in self.channels:
                return False

        if self.tags:
            notif_tags = notification.metadata.get("tags", [])
            if not any(tag in notif_tags for tag in self.tags):
                return False

        return True


@dataclass
class Subscription:
    """A notification subscription."""

    subscription_id: str
    subscriber: str
    channel: str
    filter_criteria: SubscriptionFilter | None = None
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class NotificationHandler(ABC):
    """Abstract base class for notification handlers."""

    @abstractmethod
    async def send(self, notification: Notification) -> NotificationResult:
        """Send a notification."""
        pass

    @abstractmethod
    def can_handle(self, notification: Notification) -> bool:
        """Check if handler can process notification."""
        pass


class ConsoleHandler(NotificationHandler):
    """Handler that prints notifications to console."""

    def __init__(self, format_string: str | None = None) -> None:
        """Initialize the handler."""
        self.format_string = format_string or "[{priority}] {title}: {message}"

    async def send(self, notification: Notification) -> NotificationResult:
        """Print notification to console."""
        output = self.format_string.format(
            priority=notification.priority.value.upper(),
            title=notification.title,
            message=notification.message,
            type=notification.notification_type.value,
        )
        print(output)

        return NotificationResult(
            notification_id=notification.notification_id,
            success=True,
            channel="console",
        )

    def can_handle(self, notification: Notification) -> bool:
        """Console handler can handle all notifications."""
        return True


class WebhookHandler(NotificationHandler):
    """Handler that sends notifications to webhooks."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        method: str = "POST",
    ) -> None:
        """Initialize the handler."""
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
        self.method = method

    async def send(self, notification: Notification) -> NotificationResult:
        """Send notification to webhook."""
        import json

        payload = {
            "id": notification.notification_id,
            "title": notification.title,
            "message": notification.message,
            "type": notification.notification_type.value,
            "priority": notification.priority.value,
            "timestamp": notification.created_at.isoformat(),
            "metadata": notification.metadata,
        }

        try:
            # Using aiohttp would be better, but keeping it simple
            import urllib.request

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.url,
                data=data,
                headers=self.headers,
                method=self.method,
            )

            # Note: In production, use async HTTP library
            with urllib.request.urlopen(req, timeout=10) as response:
                response_data = response.read()

            return NotificationResult(
                notification_id=notification.notification_id,
                success=True,
                channel="webhook",
                response=response_data.decode(),
            )
        except Exception as exc:
            return NotificationResult(
                notification_id=notification.notification_id,
                success=False,
                channel="webhook",
                error=str(exc),
            )

    def can_handle(self, notification: Notification) -> bool:
        """Webhook handler can handle all notifications."""
        return True


class FileHandler(NotificationHandler):
    """Handler that writes notifications to a file."""

    def __init__(self, file_path: str, append: bool = True) -> None:
        """Initialize the handler."""
        self.file_path = file_path
        self.append = append

    async def send(self, notification: Notification) -> NotificationResult:
        """Write notification to file."""
        mode = "a" if self.append else "w"
        line = (
            f"[{notification.created_at.isoformat()}] "
            f"[{notification.priority.value.upper()}] "
            f"{notification.title}: {notification.message}\n"
        )

        try:
            with open(self.file_path, mode) as file_handle:
                file_handle.write(line)

            return NotificationResult(
                notification_id=notification.notification_id,
                success=True,
                channel="file",
            )
        except Exception as exc:
            return NotificationResult(
                notification_id=notification.notification_id,
                success=False,
                channel="file",
                error=str(exc),
            )

    def can_handle(self, notification: Notification) -> bool:
        """File handler can handle all notifications."""
        return True


class CallbackHandler(NotificationHandler):
    """Handler that calls a callback function."""

    def __init__(
        self,
        callback: Callable[[Notification], Any],
        is_async: bool = False,
    ) -> None:
        """Initialize the handler."""
        self.callback = callback
        self.is_async = is_async

    async def send(self, notification: Notification) -> NotificationResult:
        """Call the callback with notification."""
        try:
            if self.is_async:
                result = await self.callback(notification)
            else:
                result = self.callback(notification)

            return NotificationResult(
                notification_id=notification.notification_id,
                success=True,
                channel="callback",
                response=result,
            )
        except Exception as exc:
            return NotificationResult(
                notification_id=notification.notification_id,
                success=False,
                channel="callback",
                error=str(exc),
            )

    def can_handle(self, notification: Notification) -> bool:
        """Callback handler can handle all notifications."""
        return True


class NotificationQueue:
    """Queue for pending notifications."""

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize the queue."""
        self.max_size = max_size
        self._queue: asyncio.Queue[Notification] = asyncio.Queue(maxsize=max_size)
        self._pending: list[Notification] = []

    async def enqueue(self, notification: Notification) -> bool:
        """Add notification to queue."""
        try:
            await asyncio.wait_for(
                self._queue.put(notification),
                timeout=1.0,
            )
            return True
        except asyncio.TimeoutError:
            self._pending.append(notification)
            return False

    async def dequeue(self) -> Notification | None:
        """Get next notification from queue."""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    def size(self) -> int:
        """Get queue size."""
        return self._queue.qsize() + len(self._pending)


class NotificationManager:
    """Main class for managing notifications."""

    def __init__(self) -> None:
        """Initialize the notification manager."""
        self._handlers: dict[str, NotificationHandler] = {}
        self._subscriptions: dict[str, Subscription] = {}
        self._queue = NotificationQueue()
        self._notification_count = 0
        self._history: list[Notification] = []
        self._max_history = 100
        self._running = False
        self._processor_task: asyncio.Task[None] | None = None

    def register_handler(
        self,
        channel: str,
        handler: NotificationHandler,
    ) -> None:
        """Register a notification handler.

        Args:
            channel: Channel name.
            handler: Handler instance.
        """
        self._handlers[channel] = handler

    def unregister_handler(self, channel: str) -> bool:
        """Unregister a handler.

        Args:
            channel: Channel name.

        Returns:
            True if handler was removed.
        """
        if channel in self._handlers:
            del self._handlers[channel]
            return True
        return False

    def subscribe(
        self,
        subscriber: str,
        channel: str,
        filter_criteria: SubscriptionFilter | None = None,
    ) -> Subscription:
        """Subscribe to notifications.

        Args:
            subscriber: Subscriber identifier.
            channel: Channel to subscribe to.
            filter_criteria: Optional filter criteria.

        Returns:
            Subscription instance.
        """
        subscription_id = f"sub_{self._notification_count}_{subscriber}"
        self._notification_count += 1

        subscription = Subscription(
            subscription_id=subscription_id,
            subscriber=subscriber,
            channel=channel,
            filter_criteria=filter_criteria,
        )

        self._subscriptions[subscription_id] = subscription
        return subscription

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from notifications.

        Args:
            subscription_id: Subscription ID.

        Returns:
            True if unsubscribed.
        """
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            return True
        return False

    async def notify(
        self,
        message: str,
        title: str = "Notification",
        notification_type: NotificationType = NotificationType.INFO,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        channel: str | None = None,
        recipient: str | None = None,
        **metadata: Any,
    ) -> NotificationResult | list[NotificationResult]:
        """Send a notification.

        Args:
            message: Notification message.
            title: Notification title.
            notification_type: Type of notification.
            priority: Notification priority.
            channel: Target channel.
            recipient: Target recipient.
            **metadata: Additional metadata.

        Returns:
            NotificationResult or list of results.
        """
        notification = self._create_notification(
            message=message,
            title=title,
            notification_type=notification_type,
            priority=priority,
            channel=channel,
            recipient=recipient,
            metadata=metadata,
        )

        # Add to history
        self._add_to_history(notification)

        # Send to specified channel or all matching subscriptions
        if channel:
            return await self._send_to_channel(notification, channel)
        else:
            return await self._send_to_subscriptions(notification)

    async def notify_async(
        self,
        message: str,
        title: str = "Notification",
        **kwargs: Any,
    ) -> None:
        """Queue a notification for async processing.

        Args:
            message: Notification message.
            title: Notification title.
            **kwargs: Additional options.
        """
        notification = self._create_notification(
            message=message,
            title=title,
            **kwargs,
        )
        await self._queue.enqueue(notification)

    def _create_notification(
        self,
        message: str,
        title: str = "Notification",
        notification_type: NotificationType = NotificationType.INFO,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        channel: str | None = None,
        recipient: str | None = None,
        metadata: dict[str, Any] | None = None,
        **extra: Any,
    ) -> Notification:
        """Create a notification."""
        self._notification_count += 1
        notification_id = f"notif_{self._notification_count}"

        merged_metadata = metadata or {}
        merged_metadata.update(extra)

        return Notification(
            notification_id=notification_id,
            title=title,
            message=message,
            notification_type=notification_type,
            priority=priority,
            channel=channel,
            recipient=recipient,
            metadata=merged_metadata,
        )

    async def _send_to_channel(
        self, notification: Notification, channel: str
    ) -> NotificationResult:
        """Send notification to a specific channel."""
        handler = self._handlers.get(channel)
        if not handler:
            return NotificationResult(
                notification_id=notification.notification_id,
                success=False,
                channel=channel,
                error=f"No handler registered for channel: {channel}",
            )

        notification.status = NotificationStatus.PENDING
        result = await handler.send(notification)

        if result.success:
            notification.status = NotificationStatus.SENT
            notification.sent_at = datetime.now()
        else:
            notification.status = NotificationStatus.FAILED

        return result

    async def _send_to_subscriptions(self, notification: Notification) -> list[NotificationResult]:
        """Send notification to all matching subscriptions."""
        results: list[NotificationResult] = []

        for subscription in self._subscriptions.values():
            if not subscription.active:
                continue

            # Check filter
            if subscription.filter_criteria:
                if not subscription.filter_criteria.matches(notification):
                    continue

            # Send to subscription channel
            result = await self._send_to_channel(notification, subscription.channel)
            results.append(result)

        return results

    def _add_to_history(self, notification: Notification) -> None:
        """Add notification to history."""
        self._history.append(notification)
        if len(self._history) > self._max_history:
            self._history.pop(0)

    async def start_processor(self) -> None:
        """Start the async notification processor."""
        if self._running:
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_queue())

    async def stop_processor(self) -> None:
        """Stop the async notification processor."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

    async def _process_queue(self) -> None:
        """Process queued notifications."""
        while self._running:
            notification = await self._queue.dequeue()
            if notification:
                if notification.channel:
                    await self._send_to_channel(notification, notification.channel)
                else:
                    await self._send_to_subscriptions(notification)
            await asyncio.sleep(0.1)

    def get_history(
        self,
        limit: int = 50,
        notification_type: NotificationType | None = None,
    ) -> list[Notification]:
        """Get notification history.

        Args:
            limit: Maximum notifications to return.
            notification_type: Filter by type.

        Returns:
            List of notifications.
        """
        history = self._history[-limit:]

        if notification_type:
            history = [n for n in history if n.notification_type == notification_type]

        return history

    def get_stats(self) -> dict[str, Any]:
        """Get notification statistics.

        Returns:
            Statistics dictionary.
        """
        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}

        for notification in self._history:
            ntype = notification.notification_type.value
            by_type[ntype] = by_type.get(ntype, 0) + 1

            status = notification.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_notifications": self._notification_count,
            "registered_handlers": list(self._handlers.keys()),
            "active_subscriptions": sum(1 for s in self._subscriptions.values() if s.active),
            "queue_size": self._queue.size(),
            "history_size": len(self._history),
            "by_type": by_type,
            "by_status": by_status,
        }


# Global instance
_notification_manager: NotificationManager | None = None


def get_notification_manager() -> NotificationManager:
    """Get the global notification manager."""
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager()
    return _notification_manager


def reset_notification_manager() -> None:
    """Reset the global notification manager."""
    global _notification_manager
    _notification_manager = None


def create_notification_manager() -> NotificationManager:
    """Create a new notification manager.

    Returns:
        New NotificationManager instance.
    """
    return NotificationManager()


async def send_notification(
    message: str,
    title: str = "Notification",
    channel: str | None = None,
    **kwargs: Any,
) -> NotificationResult | list[NotificationResult]:
    """Send a notification.

    Args:
        message: Notification message.
        title: Notification title.
        channel: Target channel.
        **kwargs: Additional options.

    Returns:
        NotificationResult or list of results.
    """
    return await get_notification_manager().notify(
        message=message,
        title=title,
        channel=channel,
        **kwargs,
    )


def create_console_handler(format_string: str | None = None) -> ConsoleHandler:
    """Create a console handler.

    Args:
        format_string: Output format string.

    Returns:
        ConsoleHandler instance.
    """
    return ConsoleHandler(format_string)


def create_webhook_handler(url: str, **kwargs: Any) -> WebhookHandler:
    """Create a webhook handler.

    Args:
        url: Webhook URL.
        **kwargs: Additional options.

    Returns:
        WebhookHandler instance.
    """
    return WebhookHandler(url, **kwargs)


def create_file_handler(file_path: str, append: bool = True) -> FileHandler:
    """Create a file handler.

    Args:
        file_path: Output file path.
        append: Whether to append to file.

    Returns:
        FileHandler instance.
    """
    return FileHandler(file_path, append)


def create_callback_handler(
    callback: Callable[[Notification], Any],
    is_async: bool = False,
) -> CallbackHandler:
    """Create a callback handler.

    Args:
        callback: Callback function.
        is_async: Whether callback is async.

    Returns:
        CallbackHandler instance.
    """
    return CallbackHandler(callback, is_async)


def create_subscription_filter(
    notification_types: list[NotificationType] | None = None,
    priorities: list[NotificationPriority] | None = None,
    **kwargs: Any,
) -> SubscriptionFilter:
    """Create a subscription filter.

    Args:
        notification_types: Types to filter.
        priorities: Priorities to filter.
        **kwargs: Additional filter options.

    Returns:
        SubscriptionFilter instance.
    """
    return SubscriptionFilter(
        notification_types=notification_types,
        priorities=priorities,
        **kwargs,
    )
