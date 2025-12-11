"""Cost tracking for explainer operations.

This module provides utilities for tracking API costs, token usage,
and resource consumption across explainer operations.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable


class CostType(Enum):
    """Type of cost being tracked."""

    API_CALL = auto()  # API call cost
    TOKEN_INPUT = auto()  # Input token cost
    TOKEN_OUTPUT = auto()  # Output token cost
    COMPUTE = auto()  # Compute/processing cost
    STORAGE = auto()  # Storage cost
    NETWORK = auto()  # Network transfer cost
    CACHE_HIT = auto()  # Cache hit (usually free/cheap)
    CACHE_MISS = auto()  # Cache miss (full cost)


class CostUnit(Enum):
    """Unit of cost measurement."""

    USD = "usd"
    TOKENS = "tokens"
    CALLS = "calls"
    BYTES = "bytes"
    MILLISECONDS = "ms"
    CREDITS = "credits"


@dataclass
class PricingTier:
    """Pricing tier for a model or service."""

    name: str
    input_price_per_1k: float = 0.0  # Price per 1000 input tokens
    output_price_per_1k: float = 0.0  # Price per 1000 output tokens
    base_price_per_call: float = 0.0  # Base price per API call
    free_tier_limit: int = 0  # Number of free calls
    rate_limit: int = 0  # Calls per minute limit
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CostEntry:
    """A single cost entry."""

    cost_type: CostType
    amount: float
    unit: CostUnit
    model: str = ""
    operation: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        """Get cost in USD if available."""
        if self.unit == CostUnit.USD:
            return self.amount
        return self.metadata.get("cost_usd", 0.0)


@dataclass
class TokenUsage:
    """Token usage for an operation."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0  # Tokens served from cache

    def __post_init__(self) -> None:
        """Calculate total if not provided."""
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens

    def add(self, other: "TokenUsage") -> "TokenUsage":
        """Add another token usage."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


@dataclass
class CostSummary:
    """Summary of costs for a period."""

    total_cost_usd: float = 0.0
    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    total_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    by_model: dict[str, float] = field(default_factory=dict)
    by_operation: dict[str, float] = field(default_factory=dict)
    by_type: dict[CostType, float] = field(default_factory=dict)
    start_time: datetime | None = None
    end_time: datetime | None = None
    entries: list[CostEntry] = field(default_factory=list)

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def cost_per_call(self) -> float:
        """Calculate average cost per call."""
        return self.total_cost_usd / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def cost_per_1k_tokens(self) -> float:
        """Calculate cost per 1000 tokens."""
        if self.total_tokens.total_tokens == 0:
            return 0.0
        return (self.total_cost_usd / self.total_tokens.total_tokens) * 1000


@dataclass
class Budget:
    """Budget configuration."""

    daily_limit_usd: float = 0.0
    monthly_limit_usd: float = 0.0
    token_limit_daily: int = 0
    call_limit_daily: int = 0
    alert_threshold: float = 0.8  # Alert when reaching 80%
    hard_limit: bool = False  # If True, block when limit reached


@dataclass
class BudgetStatus:
    """Current budget status."""

    budget: Budget
    daily_spent_usd: float = 0.0
    monthly_spent_usd: float = 0.0
    daily_tokens_used: int = 0
    daily_calls_made: int = 0
    is_over_budget: bool = False
    is_near_limit: bool = False
    remaining_daily_usd: float = 0.0
    remaining_monthly_usd: float = 0.0
    message: str = ""

    def check(self) -> None:
        """Check budget status and update flags."""
        if self.budget.daily_limit_usd > 0:
            self.remaining_daily_usd = self.budget.daily_limit_usd - self.daily_spent_usd
            daily_ratio = self.daily_spent_usd / self.budget.daily_limit_usd
            if daily_ratio >= 1.0:
                self.is_over_budget = True
                self.message = "Daily budget exceeded"
            elif daily_ratio >= self.budget.alert_threshold:
                self.is_near_limit = True
                self.message = f"Approaching daily limit ({daily_ratio:.0%})"

        if self.budget.monthly_limit_usd > 0:
            self.remaining_monthly_usd = self.budget.monthly_limit_usd - self.monthly_spent_usd
            monthly_ratio = self.monthly_spent_usd / self.budget.monthly_limit_usd
            if monthly_ratio >= 1.0:
                self.is_over_budget = True
                self.message = "Monthly budget exceeded"
            elif monthly_ratio >= self.budget.alert_threshold:
                self.is_near_limit = True
                self.message = f"Approaching monthly limit ({monthly_ratio:.0%})"


class CostBackend(ABC):
    """Abstract backend for storing cost data."""

    @abstractmethod
    def record(self, entry: CostEntry) -> None:
        """Record a cost entry."""
        ...

    @abstractmethod
    def get_entries(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        model: str | None = None,
        operation: str | None = None,
    ) -> list[CostEntry]:
        """Get cost entries with optional filters."""
        ...

    @abstractmethod
    def get_summary(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> CostSummary:
        """Get cost summary for a period."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored data."""
        ...


class InMemoryCostBackend(CostBackend):
    """In-memory cost tracking backend."""

    def __init__(self, max_entries: int = 10000):
        """Initialize backend."""
        self._entries: list[CostEntry] = []
        self._max_entries = max_entries
        self._lock = threading.Lock()

    def record(self, entry: CostEntry) -> None:
        """Record a cost entry."""
        with self._lock:
            self._entries.append(entry)
            # Trim old entries if needed
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]

    def get_entries(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        model: str | None = None,
        operation: str | None = None,
    ) -> list[CostEntry]:
        """Get cost entries with optional filters."""
        with self._lock:
            entries = self._entries.copy()

        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]
        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]
        if model:
            entries = [e for e in entries if e.model == model]
        if operation:
            entries = [e for e in entries if e.operation == operation]

        return entries

    def get_summary(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> CostSummary:
        """Get cost summary for a period."""
        entries = self.get_entries(start_time, end_time)

        summary = CostSummary(start_time=start_time, end_time=end_time, entries=entries)

        for entry in entries:
            # Total cost
            summary.total_cost_usd += entry.cost_usd

            # By model
            if entry.model:
                summary.by_model[entry.model] = (
                    summary.by_model.get(entry.model, 0.0) + entry.cost_usd
                )

            # By operation
            if entry.operation:
                summary.by_operation[entry.operation] = (
                    summary.by_operation.get(entry.operation, 0.0) + entry.cost_usd
                )

            # By type
            summary.by_type[entry.cost_type] = (
                summary.by_type.get(entry.cost_type, 0.0) + entry.cost_usd
            )

            # Track tokens
            if entry.cost_type == CostType.TOKEN_INPUT:
                summary.total_tokens.input_tokens += int(entry.amount)
            elif entry.cost_type == CostType.TOKEN_OUTPUT:
                summary.total_tokens.output_tokens += int(entry.amount)

            # Track calls
            if entry.cost_type == CostType.API_CALL:
                summary.total_calls += 1

            # Track cache
            if entry.cost_type == CostType.CACHE_HIT:
                summary.cache_hits += 1
            elif entry.cost_type == CostType.CACHE_MISS:
                summary.cache_misses += 1

        summary.total_tokens.total_tokens = (
            summary.total_tokens.input_tokens + summary.total_tokens.output_tokens
        )

        return summary

    def clear(self) -> None:
        """Clear all stored data."""
        with self._lock:
            self._entries.clear()


class CostTracker:
    """Main cost tracker for explainer operations."""

    def __init__(
        self,
        backend: CostBackend | None = None,
        budget: Budget | None = None,
    ):
        """Initialize cost tracker."""
        self._backend = backend or InMemoryCostBackend()
        self._budget = budget
        self._pricing: dict[str, PricingTier] = {}
        self._alert_callbacks: list[Callable[[BudgetStatus], None]] = []
        self._lock = threading.Lock()

        # Initialize with common pricing tiers
        self._init_default_pricing()

    def _init_default_pricing(self) -> None:
        """Initialize default pricing tiers."""
        self._pricing["gpt-4"] = PricingTier(
            name="gpt-4",
            input_price_per_1k=0.03,
            output_price_per_1k=0.06,
        )
        self._pricing["gpt-4-turbo"] = PricingTier(
            name="gpt-4-turbo",
            input_price_per_1k=0.01,
            output_price_per_1k=0.03,
        )
        self._pricing["gpt-3.5-turbo"] = PricingTier(
            name="gpt-3.5-turbo",
            input_price_per_1k=0.0015,
            output_price_per_1k=0.002,
        )
        self._pricing["claude-3-opus"] = PricingTier(
            name="claude-3-opus",
            input_price_per_1k=0.015,
            output_price_per_1k=0.075,
        )
        self._pricing["claude-3-sonnet"] = PricingTier(
            name="claude-3-sonnet",
            input_price_per_1k=0.003,
            output_price_per_1k=0.015,
        )
        self._pricing["claude-3-haiku"] = PricingTier(
            name="claude-3-haiku",
            input_price_per_1k=0.00025,
            output_price_per_1k=0.00125,
        )

    def set_pricing(self, model: str, tier: PricingTier) -> None:
        """Set pricing for a model."""
        self._pricing[model] = tier

    def get_pricing(self, model: str) -> PricingTier | None:
        """Get pricing for a model."""
        return self._pricing.get(model)

    def set_budget(self, budget: Budget) -> None:
        """Set budget configuration."""
        self._budget = budget

    def add_alert_callback(self, callback: Callable[[BudgetStatus], None]) -> None:
        """Add a callback for budget alerts."""
        self._alert_callbacks.append(callback)

    def track_tokens(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Track token usage and return cost in USD."""
        pricing = self._pricing.get(model)
        if not pricing:
            # Default pricing if model not found
            pricing = PricingTier(
                name=model,
                input_price_per_1k=0.001,
                output_price_per_1k=0.002,
            )

        input_cost = (input_tokens / 1000) * pricing.input_price_per_1k
        output_cost = (output_tokens / 1000) * pricing.output_price_per_1k
        total_cost = input_cost + output_cost

        # Record input tokens
        self._backend.record(
            CostEntry(
                cost_type=CostType.TOKEN_INPUT,
                amount=input_tokens,
                unit=CostUnit.TOKENS,
                model=model,
                operation=operation,
                metadata={"cost_usd": input_cost, **(metadata or {})},
            )
        )

        # Record output tokens
        self._backend.record(
            CostEntry(
                cost_type=CostType.TOKEN_OUTPUT,
                amount=output_tokens,
                unit=CostUnit.TOKENS,
                model=model,
                operation=operation,
                metadata={"cost_usd": output_cost, **(metadata or {})},
            )
        )

        self._check_budget()
        return total_cost

    def track_api_call(
        self,
        model: str,
        operation: str = "",
        cost_usd: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Track an API call."""
        self._backend.record(
            CostEntry(
                cost_type=CostType.API_CALL,
                amount=1,
                unit=CostUnit.CALLS,
                model=model,
                operation=operation,
                metadata={"cost_usd": cost_usd, **(metadata or {})},
            )
        )
        self._check_budget()

    def track_cache_hit(
        self,
        model: str = "",
        operation: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Track a cache hit."""
        self._backend.record(
            CostEntry(
                cost_type=CostType.CACHE_HIT,
                amount=1,
                unit=CostUnit.CALLS,
                model=model,
                operation=operation,
                metadata=metadata or {},
            )
        )

    def track_cache_miss(
        self,
        model: str = "",
        operation: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Track a cache miss."""
        self._backend.record(
            CostEntry(
                cost_type=CostType.CACHE_MISS,
                amount=1,
                unit=CostUnit.CALLS,
                model=model,
                operation=operation,
                metadata=metadata or {},
            )
        )

    def track_custom(
        self,
        cost_type: CostType,
        amount: float,
        unit: CostUnit,
        model: str = "",
        operation: str = "",
        cost_usd: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Track a custom cost entry."""
        self._backend.record(
            CostEntry(
                cost_type=cost_type,
                amount=amount,
                unit=unit,
                model=model,
                operation=operation,
                metadata={"cost_usd": cost_usd, **(metadata or {})},
            )
        )
        self._check_budget()

    def get_summary(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> CostSummary:
        """Get cost summary for a period."""
        return self._backend.get_summary(start_time, end_time)

    def get_daily_summary(self) -> CostSummary:
        """Get today's cost summary."""
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        return self.get_summary(start_time=today)

    def get_monthly_summary(self) -> CostSummary:
        """Get this month's cost summary."""
        today = datetime.utcnow()
        month_start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return self.get_summary(start_time=month_start)

    def get_budget_status(self) -> BudgetStatus | None:
        """Get current budget status."""
        if not self._budget:
            return None

        daily = self.get_daily_summary()
        monthly = self.get_monthly_summary()

        status = BudgetStatus(
            budget=self._budget,
            daily_spent_usd=daily.total_cost_usd,
            monthly_spent_usd=monthly.total_cost_usd,
            daily_tokens_used=daily.total_tokens.total_tokens,
            daily_calls_made=daily.total_calls,
        )
        status.check()
        return status

    def _check_budget(self) -> None:
        """Check budget and trigger alerts if needed."""
        status = self.get_budget_status()
        if status and (status.is_over_budget or status.is_near_limit):
            for callback in self._alert_callbacks:
                try:
                    callback(status)
                except Exception:
                    pass  # Don't fail on callback errors

    def is_within_budget(self) -> bool:
        """Check if currently within budget."""
        status = self.get_budget_status()
        if status is None:
            return True  # No budget set
        return not status.is_over_budget

    def should_block(self) -> bool:
        """Check if operations should be blocked due to budget."""
        if not self._budget:
            return False
        status = self.get_budget_status()
        return status is not None and status.is_over_budget and self._budget.hard_limit

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost for a potential operation."""
        pricing = self._pricing.get(model)
        if not pricing:
            return 0.0

        input_cost = (input_tokens / 1000) * pricing.input_price_per_1k
        output_cost = (output_tokens / 1000) * pricing.output_price_per_1k
        return input_cost + output_cost

    def clear(self) -> None:
        """Clear all tracked data."""
        self._backend.clear()


class CostContext:
    """Context manager for tracking costs of a block of code."""

    def __init__(
        self,
        tracker: CostTracker,
        operation: str = "",
        model: str = "",
    ):
        """Initialize context."""
        self.tracker = tracker
        self.operation = operation
        self.model = model
        self._start_summary: CostSummary | None = None
        self._start_time: float = 0.0

    def __enter__(self) -> "CostContext":
        """Enter context."""
        self._start_summary = self.tracker.get_summary()
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context."""
        pass

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - self._start_time) * 1000

    @property
    def cost_usd(self) -> float:
        """Get cost incurred during this context."""
        current = self.tracker.get_summary()
        if self._start_summary:
            return current.total_cost_usd - self._start_summary.total_cost_usd
        return current.total_cost_usd


def track_cost(
    tracker: CostTracker,
    operation: str = "",
    model: str = "",
) -> Callable:
    """Decorator to track cost of a function."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with CostContext(tracker, operation, model):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class CostReporter:
    """Generate cost reports."""

    def __init__(self, tracker: CostTracker):
        """Initialize reporter."""
        self.tracker = tracker

    def daily_report(self) -> dict[str, Any]:
        """Generate daily report."""
        summary = self.tracker.get_daily_summary()
        return self._format_report(summary, "Daily")

    def monthly_report(self) -> dict[str, Any]:
        """Generate monthly report."""
        summary = self.tracker.get_monthly_summary()
        return self._format_report(summary, "Monthly")

    def custom_report(
        self,
        start_time: datetime,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Generate custom period report."""
        summary = self.tracker.get_summary(start_time, end_time)
        return self._format_report(summary, "Custom")

    def _format_report(self, summary: CostSummary, period: str) -> dict[str, Any]:
        """Format a cost summary into a report."""
        return {
            "period": period,
            "total_cost_usd": round(summary.total_cost_usd, 4),
            "total_calls": summary.total_calls,
            "total_tokens": {
                "input": summary.total_tokens.input_tokens,
                "output": summary.total_tokens.output_tokens,
                "total": summary.total_tokens.total_tokens,
            },
            "cache": {
                "hits": summary.cache_hits,
                "misses": summary.cache_misses,
                "hit_rate": round(summary.cache_hit_rate, 2),
            },
            "cost_per_call": round(summary.cost_per_call, 6),
            "cost_per_1k_tokens": round(summary.cost_per_1k_tokens, 6),
            "by_model": {k: round(v, 4) for k, v in summary.by_model.items()},
            "by_operation": {k: round(v, 4) for k, v in summary.by_operation.items()},
        }

    def savings_report(self) -> dict[str, Any]:
        """Generate report showing savings from caching."""
        summary = self.tracker.get_summary()

        # Estimate what cache hits would have cost
        if summary.cache_hits > 0 and summary.cache_misses > 0:
            avg_cost_per_miss = (
                summary.total_cost_usd / summary.cache_misses if summary.cache_misses > 0 else 0
            )
            estimated_savings = summary.cache_hits * avg_cost_per_miss
        else:
            estimated_savings = 0.0

        return {
            "cache_hits": summary.cache_hits,
            "cache_misses": summary.cache_misses,
            "cache_hit_rate": round(summary.cache_hit_rate, 2),
            "estimated_savings_usd": round(estimated_savings, 4),
            "actual_cost_usd": round(summary.total_cost_usd, 4),
            "potential_cost_usd": round(summary.total_cost_usd + estimated_savings, 4),
        }


# Convenience functions
def create_cost_tracker(
    daily_limit_usd: float = 0.0,
    monthly_limit_usd: float = 0.0,
) -> CostTracker:
    """Create a cost tracker with optional budget limits."""
    budget = None
    if daily_limit_usd > 0 or monthly_limit_usd > 0:
        budget = Budget(
            daily_limit_usd=daily_limit_usd,
            monthly_limit_usd=monthly_limit_usd,
        )
    return CostTracker(budget=budget)


# Singleton tracker
_default_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """Get or create the default cost tracker."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = CostTracker()
    return _default_tracker


def reset_cost_tracker() -> None:
    """Reset the default cost tracker."""
    global _default_tracker
    if _default_tracker is not None:
        _default_tracker.clear()
    _default_tracker = None


__all__ = [
    # Enums
    "CostType",
    "CostUnit",
    # Data classes
    "PricingTier",
    "CostEntry",
    "TokenUsage",
    "CostSummary",
    "Budget",
    "BudgetStatus",
    # Backend
    "CostBackend",
    "InMemoryCostBackend",
    # Core classes
    "CostTracker",
    "CostContext",
    "CostReporter",
    # Decorator
    "track_cost",
    # Functions
    "create_cost_tracker",
    "get_cost_tracker",
    "reset_cost_tracker",
]
