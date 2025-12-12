"""User feedback collection system for code explanations.

This module provides tools for collecting, storing, and analyzing user feedback
on code explanations to improve quality over time.
"""

import hashlib
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .types import ExplanationResult


# =============================================================================
# Enums and Type Definitions
# =============================================================================


class FeedbackType(str, Enum):
    """Types of feedback that can be collected."""

    RATING = "rating"  # Numeric rating
    THUMBS = "thumbs"  # Thumbs up/down
    CHOICE = "choice"  # Multiple choice
    TEXT = "text"  # Free-form text
    CORRECTION = "correction"  # User correction
    REPORT = "report"  # Issue report


class FeedbackCategory(str, Enum):
    """Categories for feedback."""

    ACCURACY = "accuracy"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    HELPFULNESS = "helpfulness"
    TECHNICAL_DEPTH = "technical_depth"
    LENGTH = "length"
    OTHER = "other"


class FeedbackSentiment(str, Enum):
    """Sentiment of feedback."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


class FeedbackStatus(str, Enum):
    """Status of feedback processing."""

    PENDING = "pending"
    REVIEWED = "reviewed"
    ACTED_UPON = "acted_upon"
    DISMISSED = "dismissed"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FeedbackItem:
    """A single piece of feedback."""

    id: str
    type: FeedbackType
    category: FeedbackCategory
    value: Any  # Rating value, text, etc.
    sentiment: FeedbackSentiment = FeedbackSentiment.NEUTRAL
    status: FeedbackStatus = FeedbackStatus.PENDING
    explanation_id: Optional[str] = None
    element_name: Optional[str] = None
    element_type: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "category": self.category.value,
            "value": self.value,
            "sentiment": self.sentiment.value,
            "status": self.status.value,
            "explanation_id": self.explanation_id,
            "element_name": self.element_name,
            "element_type": self.element_type,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class FeedbackAggregation:
    """Aggregated feedback statistics."""

    total_count: int = 0
    positive_count: int = 0
    neutral_count: int = 0
    negative_count: int = 0
    average_rating: Optional[float] = None
    rating_distribution: Dict[int, int] = field(default_factory=dict)
    category_breakdown: Dict[str, int] = field(default_factory=dict)
    recent_feedback: List[FeedbackItem] = field(default_factory=list)
    trend: str = "stable"  # improving, declining, stable
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def positive_ratio(self) -> float:
        """Calculate positive feedback ratio."""
        return self.positive_count / self.total_count if self.total_count > 0 else 0.0

    @property
    def satisfaction_score(self) -> float:
        """Calculate overall satisfaction score (0-100)."""
        if self.total_count == 0:
            return 0.0
        weighted = self.positive_count * 100 + self.neutral_count * 50 + self.negative_count * 0
        return weighted / self.total_count


@dataclass
class FeedbackReport:
    """A comprehensive feedback report."""

    period_start: str
    period_end: str
    total_feedback: int
    aggregation: FeedbackAggregation
    top_issues: List[Dict[str, Any]] = field(default_factory=list)
    improvements_suggested: List[str] = field(default_factory=list)
    element_rankings: List[Dict[str, Any]] = field(default_factory=list)
    model_performance: Dict[str, float] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period": {
                "start": self.period_start,
                "end": self.period_end,
            },
            "total_feedback": self.total_feedback,
            "satisfaction_score": self.aggregation.satisfaction_score,
            "positive_ratio": self.aggregation.positive_ratio,
            "top_issues": self.top_issues,
            "improvements_suggested": self.improvements_suggested,
            "element_rankings": self.element_rankings,
            "model_performance": self.model_performance,
            "generated_at": self.generated_at,
        }


@dataclass
class FeedbackCollectorConfig:
    """Configuration for feedback collection."""

    enabled: bool = True
    collect_ratings: bool = True
    collect_text: bool = True
    collect_corrections: bool = True
    min_rating: int = 1
    max_rating: int = 5
    require_category: bool = False
    require_comment_on_negative: bool = True
    anonymize_users: bool = True
    retention_days: int = 365
    export_format: str = "json"
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Storage Backend
# =============================================================================


class FeedbackStorage(ABC):
    """Abstract base class for feedback storage."""

    @abstractmethod
    def save(self, item: FeedbackItem) -> str:
        """Save a feedback item."""
        pass

    @abstractmethod
    def get(self, feedback_id: str) -> Optional[FeedbackItem]:
        """Get a feedback item by ID."""
        pass

    @abstractmethod
    def query(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[FeedbackItem]:
        """Query feedback items."""
        pass

    @abstractmethod
    def delete(self, feedback_id: str) -> bool:
        """Delete a feedback item."""
        pass

    @abstractmethod
    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count feedback items."""
        pass


class InMemoryFeedbackStorage(FeedbackStorage):
    """In-memory feedback storage."""

    def __init__(self):
        """Initialize storage."""
        self._items: Dict[str, FeedbackItem] = {}

    def save(self, item: FeedbackItem) -> str:
        """Save feedback item."""
        self._items[item.id] = item
        return item.id

    def get(self, feedback_id: str) -> Optional[FeedbackItem]:
        """Get feedback by ID."""
        return self._items.get(feedback_id)

    def query(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[FeedbackItem]:
        """Query feedback items."""
        items = list(self._items.values())

        if filters:
            for key, value in filters.items():
                if key == "type" and isinstance(value, FeedbackType):
                    items = [i for i in items if i.type == value]
                elif key == "category" and isinstance(value, FeedbackCategory):
                    items = [i for i in items if i.category == value]
                elif key == "sentiment" and isinstance(value, FeedbackSentiment):
                    items = [i for i in items if i.sentiment == value]
                elif key == "element_name":
                    items = [i for i in items if i.element_name == value]
                elif key == "status" and isinstance(value, FeedbackStatus):
                    items = [i for i in items if i.status == value]
                elif key == "since":
                    items = [i for i in items if i.timestamp >= value]

        # Sort by timestamp descending
        items.sort(key=lambda x: x.timestamp, reverse=True)

        return items[offset : offset + limit]

    def delete(self, feedback_id: str) -> bool:
        """Delete feedback item."""
        if feedback_id in self._items:
            del self._items[feedback_id]
            return True
        return False

    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count feedback items."""
        return len(self.query(filters, limit=10000))

    def clear(self) -> None:
        """Clear all items."""
        self._items.clear()


# =============================================================================
# Feedback Collector
# =============================================================================


class FeedbackCollector:
    """Collects and manages user feedback on explanations."""

    def __init__(
        self,
        config: Optional[FeedbackCollectorConfig] = None,
        storage: Optional[FeedbackStorage] = None,
    ):
        """Initialize feedback collector.

        Args:
            config: Collector configuration.
            storage: Optional custom storage backend.
        """
        self.config = config or FeedbackCollectorConfig()
        self.storage = storage or InMemoryFeedbackStorage()
        self._listeners: List[Callable[[FeedbackItem], None]] = []

    def collect_rating(
        self,
        explanation: ExplanationResult,
        rating: int,
        category: FeedbackCategory = FeedbackCategory.HELPFULNESS,
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> FeedbackItem:
        """Collect a rating feedback.

        Args:
            explanation: The explanation being rated.
            rating: Numeric rating (within configured range).
            category: Category of feedback.
            comment: Optional comment.
            user_id: Optional user identifier.
            session_id: Optional session identifier.

        Returns:
            Created FeedbackItem.
        """
        # Validate rating
        rating = max(self.config.min_rating, min(rating, self.config.max_rating))

        # Determine sentiment
        mid_point = (self.config.max_rating + self.config.min_rating) / 2
        if rating > mid_point + 0.5:
            sentiment = FeedbackSentiment.POSITIVE
        elif rating < mid_point - 0.5:
            sentiment = FeedbackSentiment.NEGATIVE
        else:
            sentiment = FeedbackSentiment.NEUTRAL

        feedback_id = self._generate_id(explanation, "rating")

        item = FeedbackItem(
            id=feedback_id,
            type=FeedbackType.RATING,
            category=category,
            value={"rating": rating, "comment": comment},
            sentiment=sentiment,
            explanation_id=self._get_explanation_id(explanation),
            element_name=explanation.element.name,
            element_type=explanation.element.type.value,
            user_id=self._anonymize_user(user_id) if self.config.anonymize_users else user_id,
            session_id=session_id,
        )

        self.storage.save(item)
        self._notify_listeners(item)
        return item

    def collect_thumbs(
        self,
        explanation: ExplanationResult,
        thumbs_up: bool,
        category: FeedbackCategory = FeedbackCategory.HELPFULNESS,
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> FeedbackItem:
        """Collect thumbs up/down feedback.

        Args:
            explanation: The explanation being rated.
            thumbs_up: True for thumbs up, False for thumbs down.
            category: Category of feedback.
            comment: Optional comment.
            user_id: Optional user identifier.

        Returns:
            Created FeedbackItem.
        """
        sentiment = FeedbackSentiment.POSITIVE if thumbs_up else FeedbackSentiment.NEGATIVE
        feedback_id = self._generate_id(explanation, "thumbs")

        item = FeedbackItem(
            id=feedback_id,
            type=FeedbackType.THUMBS,
            category=category,
            value={"thumbs_up": thumbs_up, "comment": comment},
            sentiment=sentiment,
            explanation_id=self._get_explanation_id(explanation),
            element_name=explanation.element.name,
            element_type=explanation.element.type.value,
            user_id=self._anonymize_user(user_id) if self.config.anonymize_users else user_id,
        )

        self.storage.save(item)
        self._notify_listeners(item)
        return item

    def collect_correction(
        self,
        explanation: ExplanationResult,
        original_text: str,
        corrected_text: str,
        reason: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> FeedbackItem:
        """Collect a user correction to an explanation.

        Args:
            explanation: The explanation being corrected.
            original_text: Original text being corrected.
            corrected_text: User's correction.
            reason: Optional reason for correction.
            user_id: Optional user identifier.

        Returns:
            Created FeedbackItem.
        """
        feedback_id = self._generate_id(explanation, "correction")

        item = FeedbackItem(
            id=feedback_id,
            type=FeedbackType.CORRECTION,
            category=FeedbackCategory.ACCURACY,
            value={
                "original": original_text,
                "corrected": corrected_text,
                "reason": reason,
            },
            sentiment=FeedbackSentiment.NEGATIVE,  # Corrections indicate issues
            explanation_id=self._get_explanation_id(explanation),
            element_name=explanation.element.name,
            element_type=explanation.element.type.value,
            user_id=self._anonymize_user(user_id) if self.config.anonymize_users else user_id,
        )

        self.storage.save(item)
        self._notify_listeners(item)
        return item

    def collect_report(
        self,
        explanation: ExplanationResult,
        issue_type: str,
        description: str,
        severity: str = "medium",
        user_id: Optional[str] = None,
    ) -> FeedbackItem:
        """Collect an issue report.

        Args:
            explanation: The explanation with the issue.
            issue_type: Type of issue (e.g., "hallucination", "outdated", "confusing").
            description: Description of the issue.
            severity: Severity level.
            user_id: Optional user identifier.

        Returns:
            Created FeedbackItem.
        """
        feedback_id = self._generate_id(explanation, "report")

        item = FeedbackItem(
            id=feedback_id,
            type=FeedbackType.REPORT,
            category=FeedbackCategory.ACCURACY,
            value={
                "issue_type": issue_type,
                "description": description,
                "severity": severity,
            },
            sentiment=FeedbackSentiment.NEGATIVE,
            explanation_id=self._get_explanation_id(explanation),
            element_name=explanation.element.name,
            element_type=explanation.element.type.value,
            user_id=self._anonymize_user(user_id) if self.config.anonymize_users else user_id,
        )

        self.storage.save(item)
        self._notify_listeners(item)
        return item

    def collect_text_feedback(
        self,
        explanation: ExplanationResult,
        text: str,
        category: FeedbackCategory = FeedbackCategory.OTHER,
        user_id: Optional[str] = None,
    ) -> FeedbackItem:
        """Collect free-form text feedback.

        Args:
            explanation: The explanation being commented on.
            text: User's feedback text.
            category: Category of feedback.
            user_id: Optional user identifier.

        Returns:
            Created FeedbackItem.
        """
        # Simple sentiment analysis
        sentiment = self._analyze_sentiment(text)
        feedback_id = self._generate_id(explanation, "text")

        item = FeedbackItem(
            id=feedback_id,
            type=FeedbackType.TEXT,
            category=category,
            value={"text": text},
            sentiment=sentiment,
            explanation_id=self._get_explanation_id(explanation),
            element_name=explanation.element.name,
            element_type=explanation.element.type.value,
            user_id=self._anonymize_user(user_id) if self.config.anonymize_users else user_id,
        )

        self.storage.save(item)
        self._notify_listeners(item)
        return item

    def add_listener(self, listener: Callable[[FeedbackItem], None]) -> None:
        """Add a feedback listener."""
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[FeedbackItem], None]) -> None:
        """Remove a feedback listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify_listeners(self, item: FeedbackItem) -> None:
        """Notify all listeners of new feedback."""
        for listener in self._listeners:
            try:
                listener(item)
            except Exception:
                pass  # Don't let listener errors propagate

    def _generate_id(self, explanation: ExplanationResult, feedback_type: str) -> str:
        """Generate a unique feedback ID."""
        data = f"{explanation.element.name}:{feedback_type}:{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _get_explanation_id(self, explanation: ExplanationResult) -> str:
        """Get or generate an explanation ID."""
        if hasattr(explanation, "id"):
            return explanation.id
        data = f"{explanation.element.name}:{explanation.explanation[:50]}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _anonymize_user(self, user_id: Optional[str]) -> Optional[str]:
        """Anonymize a user ID."""
        if not user_id:
            return None
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]

    def _analyze_sentiment(self, text: str) -> FeedbackSentiment:
        """Simple sentiment analysis on text."""
        text_lower = text.lower()

        positive_words = {
            "good",
            "great",
            "excellent",
            "helpful",
            "clear",
            "useful",
            "thanks",
            "perfect",
            "awesome",
            "love",
        }
        negative_words = {
            "bad",
            "wrong",
            "incorrect",
            "confusing",
            "unclear",
            "useless",
            "poor",
            "terrible",
            "hate",
            "awful",
        }

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return FeedbackSentiment.POSITIVE
        elif negative_count > positive_count:
            return FeedbackSentiment.NEGATIVE
        elif positive_count > 0 and negative_count > 0:
            return FeedbackSentiment.MIXED
        return FeedbackSentiment.NEUTRAL


# =============================================================================
# Feedback Analyzer
# =============================================================================


class FeedbackAnalyzer:
    """Analyzes collected feedback to generate insights."""

    def __init__(self, storage: FeedbackStorage):
        """Initialize analyzer with storage backend."""
        self.storage = storage

    def aggregate(
        self,
        filters: Optional[Dict[str, Any]] = None,
        since: Optional[datetime] = None,
    ) -> FeedbackAggregation:
        """Aggregate feedback statistics.

        Args:
            filters: Optional filters for feedback.
            since: Optional start date for aggregation.

        Returns:
            FeedbackAggregation with statistics.
        """
        if since:
            filters = filters or {}
            filters["since"] = since.isoformat()

        items = self.storage.query(filters, limit=10000)

        aggregation = FeedbackAggregation()
        aggregation.total_count = len(items)

        ratings = []
        for item in items:
            # Count by sentiment
            if item.sentiment == FeedbackSentiment.POSITIVE:
                aggregation.positive_count += 1
            elif item.sentiment == FeedbackSentiment.NEGATIVE:
                aggregation.negative_count += 1
            else:
                aggregation.neutral_count += 1

            # Count by category
            category = item.category.value
            aggregation.category_breakdown[category] = (
                aggregation.category_breakdown.get(category, 0) + 1
            )

            # Collect ratings
            if item.type == FeedbackType.RATING and isinstance(item.value, dict):
                rating = item.value.get("rating")
                if rating is not None:
                    ratings.append(rating)
                    aggregation.rating_distribution[rating] = (
                        aggregation.rating_distribution.get(rating, 0) + 1
                    )

        # Calculate average rating
        if ratings:
            aggregation.average_rating = statistics.mean(ratings)

        # Get recent feedback (last 10)
        aggregation.recent_feedback = items[:10]

        # Determine trend
        aggregation.trend = self._calculate_trend(items)

        return aggregation

    def generate_report(
        self,
        period_days: int = 30,
        filters: Optional[Dict[str, Any]] = None,
    ) -> FeedbackReport:
        """Generate a comprehensive feedback report.

        Args:
            period_days: Number of days to include in report.
            filters: Optional additional filters.

        Returns:
            FeedbackReport with analysis.
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)

        filters = filters or {}
        filters["since"] = start_date.isoformat()

        items = self.storage.query(filters, limit=10000)
        aggregation = self.aggregate(filters, start_date)

        # Find top issues
        top_issues = self._find_top_issues(items)

        # Generate improvement suggestions
        improvements = self._generate_improvements(items, aggregation)

        # Rank elements by feedback
        element_rankings = self._rank_elements(items)

        # Analyze model performance if available
        model_performance = self._analyze_model_performance(items)

        return FeedbackReport(
            period_start=start_date.isoformat(),
            period_end=end_date.isoformat(),
            total_feedback=len(items),
            aggregation=aggregation,
            top_issues=top_issues,
            improvements_suggested=improvements,
            element_rankings=element_rankings,
            model_performance=model_performance,
        )

    def get_element_feedback(self, element_name: str, limit: int = 50) -> List[FeedbackItem]:
        """Get feedback for a specific element.

        Args:
            element_name: Name of the element.
            limit: Maximum items to return.

        Returns:
            List of FeedbackItem objects.
        """
        return self.storage.query({"element_name": element_name}, limit=limit)

    def get_corrections(self, limit: int = 100) -> List[FeedbackItem]:
        """Get all user corrections.

        Args:
            limit: Maximum items to return.

        Returns:
            List of correction FeedbackItems.
        """
        return self.storage.query({"type": FeedbackType.CORRECTION}, limit=limit)

    def get_reports(
        self, status: Optional[FeedbackStatus] = None, limit: int = 100
    ) -> List[FeedbackItem]:
        """Get issue reports.

        Args:
            status: Optional status filter.
            limit: Maximum items to return.

        Returns:
            List of report FeedbackItems.
        """
        filters: Dict[str, Any] = {"type": FeedbackType.REPORT}
        if status:
            filters["status"] = status
        return self.storage.query(filters, limit=limit)

    def _calculate_trend(self, items: List[FeedbackItem]) -> str:
        """Calculate feedback trend over time."""
        if len(items) < 10:
            return "stable"

        # Split into halves
        mid = len(items) // 2
        recent = items[:mid]
        older = items[mid:]

        recent_positive = sum(1 for i in recent if i.sentiment == FeedbackSentiment.POSITIVE)
        older_positive = sum(1 for i in older if i.sentiment == FeedbackSentiment.POSITIVE)

        recent_ratio = recent_positive / len(recent) if recent else 0
        older_ratio = older_positive / len(older) if older else 0

        if recent_ratio > older_ratio + 0.1:
            return "improving"
        elif recent_ratio < older_ratio - 0.1:
            return "declining"
        return "stable"

    def _find_top_issues(self, items: List[FeedbackItem]) -> List[Dict[str, Any]]:
        """Find most common issues from feedback."""
        issue_counts: Dict[str, int] = {}

        for item in items:
            if item.sentiment == FeedbackSentiment.NEGATIVE:
                if item.type == FeedbackType.REPORT and isinstance(item.value, dict):
                    issue_type = item.value.get("issue_type", "unknown")
                    issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
                else:
                    issue_counts[item.category.value] = issue_counts.get(item.category.value, 0) + 1

        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"issue": k, "count": v} for k, v in sorted_issues[:10]]

    def _generate_improvements(
        self, items: List[FeedbackItem], aggregation: FeedbackAggregation
    ) -> List[str]:
        """Generate improvement suggestions based on feedback."""
        suggestions = []

        # Check category breakdown
        if aggregation.category_breakdown.get("clarity", 0) > aggregation.total_count * 0.2:
            suggestions.append("Focus on making explanations clearer and more concise")

        if aggregation.category_breakdown.get("accuracy", 0) > aggregation.total_count * 0.15:
            suggestions.append("Review accuracy of technical claims in explanations")

        if aggregation.category_breakdown.get("completeness", 0) > aggregation.total_count * 0.2:
            suggestions.append("Ensure explanations cover all key aspects of the code")

        # Check average rating
        if aggregation.average_rating and aggregation.average_rating < 3.0:
            suggestions.append("Overall quality needs improvement - average rating below 3.0")

        # Check negative trend
        if aggregation.trend == "declining":
            suggestions.append("User satisfaction is declining - investigate recent changes")

        return suggestions

    def _rank_elements(self, items: List[FeedbackItem]) -> List[Dict[str, Any]]:
        """Rank elements by feedback quality."""
        element_stats: Dict[str, Dict[str, Any]] = {}

        for item in items:
            name = item.element_name or "unknown"
            if name not in element_stats:
                element_stats[name] = {"positive": 0, "negative": 0, "total": 0}

            element_stats[name]["total"] += 1
            if item.sentiment == FeedbackSentiment.POSITIVE:
                element_stats[name]["positive"] += 1
            elif item.sentiment == FeedbackSentiment.NEGATIVE:
                element_stats[name]["negative"] += 1

        rankings = []
        for name, stats in element_stats.items():
            if stats["total"] >= 3:  # Minimum feedback threshold
                score = (
                    (stats["positive"] - stats["negative"]) / stats["total"]
                    if stats["total"] > 0
                    else 0
                )
                rankings.append(
                    {
                        "element": name,
                        "score": score,
                        "feedback_count": stats["total"],
                        "positive": stats["positive"],
                        "negative": stats["negative"],
                    }
                )

        rankings.sort(key=lambda x: x["score"], reverse=True)
        return rankings[:20]

    def _analyze_model_performance(self, items: List[FeedbackItem]) -> Dict[str, float]:
        """Analyze performance by model (if tracked in metadata)."""
        model_stats: Dict[str, Dict[str, int]] = {}

        for item in items:
            model = item.metadata.get("model", "unknown")
            if model not in model_stats:
                model_stats[model] = {"positive": 0, "total": 0}

            model_stats[model]["total"] += 1
            if item.sentiment == FeedbackSentiment.POSITIVE:
                model_stats[model]["positive"] += 1

        return {
            model: stats["positive"] / stats["total"] if stats["total"] > 0 else 0.0
            for model, stats in model_stats.items()
            if stats["total"] >= 5
        }


# =============================================================================
# Feedback Exporter
# =============================================================================


class FeedbackExporter:
    """Exports feedback data in various formats."""

    def __init__(self, storage: FeedbackStorage):
        """Initialize exporter with storage backend."""
        self.storage = storage

    def export_json(self, filters: Optional[Dict[str, Any]] = None, pretty: bool = True) -> str:
        """Export feedback as JSON.

        Args:
            filters: Optional filters for feedback.
            pretty: Whether to pretty-print JSON.

        Returns:
            JSON string.
        """
        import json

        items = self.storage.query(filters, limit=10000)
        data = [item.to_dict() for item in items]
        return json.dumps(data, indent=2 if pretty else None)

    def export_csv(self, filters: Optional[Dict[str, Any]] = None) -> str:
        """Export feedback as CSV.

        Args:
            filters: Optional filters for feedback.

        Returns:
            CSV string.
        """
        import csv
        import io

        items = self.storage.query(filters, limit=10000)

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(
            [
                "id",
                "type",
                "category",
                "sentiment",
                "status",
                "element_name",
                "element_type",
                "timestamp",
            ]
        )

        # Data
        for item in items:
            writer.writerow(
                [
                    item.id,
                    item.type.value,
                    item.category.value,
                    item.sentiment.value,
                    item.status.value,
                    item.element_name,
                    item.element_type,
                    item.timestamp,
                ]
            )

        return output.getvalue()

    def export_to_file(
        self,
        file_path: str,
        format: str = "json",
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Export feedback to a file.

        Args:
            file_path: Path to output file.
            format: Export format ('json' or 'csv').
            filters: Optional filters for feedback.
        """
        if format == "json":
            content = self.export_json(filters)
        elif format == "csv":
            content = self.export_csv(filters)
        else:
            raise ValueError(f"Unknown format: {format}")

        with open(file_path, "w") as f:
            f.write(content)


# =============================================================================
# Global Instance Management
# =============================================================================


_global_feedback_collector: Optional[FeedbackCollector] = None


def get_feedback_collector() -> FeedbackCollector:
    """Get the global feedback collector instance."""
    global _global_feedback_collector
    if _global_feedback_collector is None:
        _global_feedback_collector = FeedbackCollector()
    return _global_feedback_collector


def reset_feedback_collector() -> None:
    """Reset the global feedback collector."""
    global _global_feedback_collector
    _global_feedback_collector = None


def collect_rating(
    explanation: ExplanationResult,
    rating: int,
    category: FeedbackCategory = FeedbackCategory.HELPFULNESS,
    comment: Optional[str] = None,
) -> FeedbackItem:
    """Convenience function to collect a rating."""
    return get_feedback_collector().collect_rating(explanation, rating, category, comment)


def collect_thumbs(
    explanation: ExplanationResult,
    thumbs_up: bool,
    comment: Optional[str] = None,
) -> FeedbackItem:
    """Convenience function to collect thumbs feedback."""
    return get_feedback_collector().collect_thumbs(explanation, thumbs_up, comment=comment)


def create_feedback_collector(
    config: Optional[FeedbackCollectorConfig] = None,
) -> FeedbackCollector:
    """Create a new feedback collector with optional config."""
    return FeedbackCollector(config=config)
