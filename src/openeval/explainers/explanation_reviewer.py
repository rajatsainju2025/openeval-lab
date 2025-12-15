"""
Explanation Reviewer for peer review workflows.

This module provides a complete peer review system for code explanations,
including review requests, approval workflows, and suggestion tracking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4


class ReviewStatus(Enum):
    """Status of a review request."""

    DRAFT = auto()
    PENDING = auto()
    IN_REVIEW = auto()
    CHANGES_REQUESTED = auto()
    APPROVED = auto()
    REJECTED = auto()
    MERGED = auto()
    CLOSED = auto()


class ReviewPriority(Enum):
    """Priority level for review requests."""

    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    URGENT = auto()
    CRITICAL = auto()


class CommentType(Enum):
    """Types of review comments."""

    GENERAL = auto()
    SUGGESTION = auto()
    QUESTION = auto()
    PRAISE = auto()
    ISSUE = auto()
    NITPICK = auto()
    BLOCKING = auto()


class VoteType(Enum):
    """Types of review votes."""

    APPROVE = auto()
    REQUEST_CHANGES = auto()
    COMMENT_ONLY = auto()
    REJECT = auto()


@dataclass
class Reviewer:
    """Represents a reviewer."""

    id: str
    name: str
    email: str
    expertise: List[str] = field(default_factory=list)
    reviews_completed: int = 0
    average_response_time_hours: float = 24.0
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "expertise": self.expertise,
            "reviews_completed": self.reviews_completed,
            "average_response_time_hours": self.average_response_time_hours,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ReviewComment:
    """A comment on a review."""

    id: str
    review_id: str
    author_id: str
    comment_type: CommentType
    content: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    replies: List["ReviewComment"] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "review_id": self.review_id,
            "author_id": self.author_id,
            "comment_type": self.comment_type.name,
            "content": self.content,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "resolved": self.resolved,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "replies": [r.to_dict() for r in self.replies],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class ReviewSuggestion:
    """A suggested change to the explanation."""

    id: str
    review_id: str
    author_id: str
    original_text: str
    suggested_text: str
    line_start: int
    line_end: int
    rationale: str
    accepted: Optional[bool] = None
    accepted_by: Optional[str] = None
    accepted_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "review_id": self.review_id,
            "author_id": self.author_id,
            "original_text": self.original_text,
            "suggested_text": self.suggested_text,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "rationale": self.rationale,
            "accepted": self.accepted,
            "accepted_by": self.accepted_by,
            "accepted_at": self.accepted_at.isoformat() if self.accepted_at else None,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ReviewVote:
    """A vote on a review."""

    id: str
    review_id: str
    reviewer_id: str
    vote_type: VoteType
    comment: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "review_id": self.review_id,
            "reviewer_id": self.reviewer_id,
            "vote_type": self.vote_type.name,
            "comment": self.comment,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ReviewRequest:
    """A review request for an explanation."""

    id: str
    title: str
    description: str
    explanation_id: str
    explanation_content: str
    code_reference: str
    author_id: str
    status: ReviewStatus
    priority: ReviewPriority
    assigned_reviewers: List[str] = field(default_factory=list)
    required_approvals: int = 1
    labels: List[str] = field(default_factory=list)
    comments: List[ReviewComment] = field(default_factory=list)
    suggestions: List[ReviewSuggestion] = field(default_factory=list)
    votes: List[ReviewVote] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    merged_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "explanation_id": self.explanation_id,
            "explanation_content": (
                self.explanation_content[:200] + "..."
                if len(self.explanation_content) > 200
                else self.explanation_content
            ),
            "code_reference": self.code_reference,
            "author_id": self.author_id,
            "status": self.status.name,
            "priority": self.priority.name,
            "assigned_reviewers": self.assigned_reviewers,
            "required_approvals": self.required_approvals,
            "labels": self.labels,
            "comments_count": len(self.comments),
            "suggestions_count": len(self.suggestions),
            "votes_count": len(self.votes),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "merged_at": self.merged_at.isoformat() if self.merged_at else None,
        }

    def get_approval_count(self) -> int:
        """Get the number of approvals."""
        return sum(1 for v in self.votes if v.vote_type == VoteType.APPROVE)

    def is_approved(self) -> bool:
        """Check if the review has enough approvals."""
        return self.get_approval_count() >= self.required_approvals

    def has_blocking_comments(self) -> bool:
        """Check if there are unresolved blocking comments."""
        return any(c.comment_type == CommentType.BLOCKING and not c.resolved for c in self.comments)


@dataclass
class ReviewMetrics:
    """Metrics for a reviewer or review process."""

    total_reviews: int = 0
    reviews_completed: int = 0
    average_response_time_hours: float = 0.0
    average_comments_per_review: float = 0.0
    approval_rate: float = 0.0
    suggestions_accepted_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_reviews": self.total_reviews,
            "reviews_completed": self.reviews_completed,
            "average_response_time_hours": self.average_response_time_hours,
            "average_comments_per_review": self.average_comments_per_review,
            "approval_rate": self.approval_rate,
            "suggestions_accepted_rate": self.suggestions_accepted_rate,
        }


class ReviewStorage(ABC):
    """Abstract storage for review data."""

    @abstractmethod
    def save_review(self, review: ReviewRequest) -> None:
        """Save a review request."""
        pass

    @abstractmethod
    def get_review(self, review_id: str) -> Optional[ReviewRequest]:
        """Get a review by ID."""
        pass

    @abstractmethod
    def list_reviews(
        self,
        status: Optional[ReviewStatus] = None,
        author_id: Optional[str] = None,
        reviewer_id: Optional[str] = None,
    ) -> List[ReviewRequest]:
        """List reviews with optional filters."""
        pass

    @abstractmethod
    def delete_review(self, review_id: str) -> bool:
        """Delete a review."""
        pass

    @abstractmethod
    def save_reviewer(self, reviewer: Reviewer) -> None:
        """Save a reviewer."""
        pass

    @abstractmethod
    def get_reviewer(self, reviewer_id: str) -> Optional[Reviewer]:
        """Get a reviewer by ID."""
        pass

    @abstractmethod
    def list_reviewers(self) -> List[Reviewer]:
        """List all reviewers."""
        pass


class InMemoryReviewStorage(ReviewStorage):
    """In-memory storage for reviews."""

    def __init__(self):
        self.reviews: Dict[str, ReviewRequest] = {}
        self.reviewers: Dict[str, Reviewer] = {}

    def save_review(self, review: ReviewRequest) -> None:
        """Save a review request."""
        self.reviews[review.id] = review

    def get_review(self, review_id: str) -> Optional[ReviewRequest]:
        """Get a review by ID."""
        return self.reviews.get(review_id)

    def list_reviews(
        self,
        status: Optional[ReviewStatus] = None,
        author_id: Optional[str] = None,
        reviewer_id: Optional[str] = None,
    ) -> List[ReviewRequest]:
        """List reviews with optional filters."""
        reviews = list(self.reviews.values())

        if status:
            reviews = [r for r in reviews if r.status == status]
        if author_id:
            reviews = [r for r in reviews if r.author_id == author_id]
        if reviewer_id:
            reviews = [r for r in reviews if reviewer_id in r.assigned_reviewers]

        return sorted(reviews, key=lambda r: r.created_at, reverse=True)

    def delete_review(self, review_id: str) -> bool:
        """Delete a review."""
        if review_id in self.reviews:
            del self.reviews[review_id]
            return True
        return False

    def save_reviewer(self, reviewer: Reviewer) -> None:
        """Save a reviewer."""
        self.reviewers[reviewer.id] = reviewer

    def get_reviewer(self, reviewer_id: str) -> Optional[Reviewer]:
        """Get a reviewer by ID."""
        return self.reviewers.get(reviewer_id)

    def list_reviewers(self) -> List[Reviewer]:
        """List all reviewers."""
        return list(self.reviewers.values())


class ReviewerMatcher(ABC):
    """Abstract base for matching reviewers to reviews."""

    @abstractmethod
    def find_reviewers(
        self,
        review: ReviewRequest,
        available_reviewers: List[Reviewer],
        count: int = 1,
    ) -> List[Reviewer]:
        """Find suitable reviewers for a review."""
        pass


class ExpertiseBasedMatcher(ReviewerMatcher):
    """Match reviewers based on expertise."""

    def find_reviewers(
        self,
        review: ReviewRequest,
        available_reviewers: List[Reviewer],
        count: int = 1,
    ) -> List[Reviewer]:
        """Find reviewers with matching expertise."""
        # Extract keywords from review
        keywords = set()
        for label in review.labels:
            keywords.add(label.lower())
        keywords.update(review.title.lower().split())

        # Score reviewers by expertise match
        scored_reviewers = []
        for reviewer in available_reviewers:
            if reviewer.id == review.author_id:
                continue  # Skip the author

            score = sum(
                1
                for exp in reviewer.expertise
                if exp.lower() in keywords or any(k in exp.lower() for k in keywords)
            )
            scored_reviewers.append((reviewer, score))

        # Sort by score descending, then by response time
        scored_reviewers.sort(key=lambda x: (-x[1], x[0].average_response_time_hours))

        return [r for r, _ in scored_reviewers[:count]]


class LoadBalancedMatcher(ReviewerMatcher):
    """Match reviewers with load balancing."""

    def __init__(self, storage: ReviewStorage):
        self.storage = storage

    def find_reviewers(
        self,
        review: ReviewRequest,
        available_reviewers: List[Reviewer],
        count: int = 1,
    ) -> List[Reviewer]:
        """Find reviewers with the least current load."""
        # Count pending reviews per reviewer
        pending_reviews = self.storage.list_reviews(status=ReviewStatus.PENDING)
        in_review = self.storage.list_reviews(status=ReviewStatus.IN_REVIEW)

        review_counts: Dict[str, int] = {}
        for r in pending_reviews + in_review:
            for reviewer_id in r.assigned_reviewers:
                review_counts[reviewer_id] = review_counts.get(reviewer_id, 0) + 1

        # Score reviewers by current load (lower is better)
        scored_reviewers = []
        for reviewer in available_reviewers:
            if reviewer.id == review.author_id:
                continue
            load = review_counts.get(reviewer.id, 0)
            scored_reviewers.append((reviewer, load))

        # Sort by load ascending
        scored_reviewers.sort(key=lambda x: x[1])

        return [r for r, _ in scored_reviewers[:count]]


class ExplanationReviewer:
    """Main class for managing the review process."""

    def __init__(
        self,
        storage: Optional[ReviewStorage] = None,
        matcher: Optional[ReviewerMatcher] = None,
    ):
        """Initialize the reviewer system."""
        self.storage = storage or InMemoryReviewStorage()
        self.matcher = matcher or ExpertiseBasedMatcher()
        self._event_handlers: Dict[str, List[Callable]] = {}

    def register_reviewer(
        self,
        name: str,
        email: str,
        expertise: Optional[List[str]] = None,
        reviewer_id: Optional[str] = None,
    ) -> Reviewer:
        """Register a new reviewer."""
        reviewer = Reviewer(
            id=reviewer_id or str(uuid4()),
            name=name,
            email=email,
            expertise=expertise or [],
        )
        self.storage.save_reviewer(reviewer)
        self._emit_event("reviewer_registered", reviewer)
        return reviewer

    def create_review(
        self,
        title: str,
        description: str,
        explanation_id: str,
        explanation_content: str,
        code_reference: str,
        author_id: str,
        priority: ReviewPriority = ReviewPriority.NORMAL,
        labels: Optional[List[str]] = None,
        required_approvals: int = 1,
        auto_assign: bool = True,
        reviewers: Optional[List[str]] = None,
    ) -> ReviewRequest:
        """Create a new review request."""
        review = ReviewRequest(
            id=str(uuid4()),
            title=title,
            description=description,
            explanation_id=explanation_id,
            explanation_content=explanation_content,
            code_reference=code_reference,
            author_id=author_id,
            status=ReviewStatus.DRAFT,
            priority=priority,
            labels=labels or [],
            required_approvals=required_approvals,
        )

        # Assign reviewers
        if reviewers:
            review.assigned_reviewers = reviewers
        elif auto_assign:
            available = self.storage.list_reviewers()
            matched = self.matcher.find_reviewers(review, available, count=required_approvals)
            review.assigned_reviewers = [r.id for r in matched]

        self.storage.save_review(review)
        self._emit_event("review_created", review)
        return review

    def submit_review(self, review_id: str) -> ReviewRequest:
        """Submit a draft review for review."""
        review = self.storage.get_review(review_id)
        if not review:
            raise ValueError(f"Review {review_id} not found")

        if review.status != ReviewStatus.DRAFT:
            raise ValueError(f"Review is not in draft status: {review.status.name}")

        review.status = ReviewStatus.PENDING
        review.updated_at = datetime.now()
        self.storage.save_review(review)
        self._emit_event("review_submitted", review)
        return review

    def start_review(self, review_id: str, reviewer_id: str) -> ReviewRequest:
        """Start reviewing a review request."""
        review = self.storage.get_review(review_id)
        if not review:
            raise ValueError(f"Review {review_id} not found")

        if reviewer_id not in review.assigned_reviewers:
            raise ValueError(f"Reviewer {reviewer_id} is not assigned to this review")

        review.status = ReviewStatus.IN_REVIEW
        review.updated_at = datetime.now()
        self.storage.save_review(review)
        self._emit_event("review_started", review, reviewer_id)
        return review

    def add_comment(
        self,
        review_id: str,
        author_id: str,
        content: str,
        comment_type: CommentType = CommentType.GENERAL,
        line_start: Optional[int] = None,
        line_end: Optional[int] = None,
    ) -> ReviewComment:
        """Add a comment to a review."""
        review = self.storage.get_review(review_id)
        if not review:
            raise ValueError(f"Review {review_id} not found")

        comment = ReviewComment(
            id=str(uuid4()),
            review_id=review_id,
            author_id=author_id,
            comment_type=comment_type,
            content=content,
            line_start=line_start,
            line_end=line_end,
        )

        review.comments.append(comment)
        review.updated_at = datetime.now()
        self.storage.save_review(review)
        self._emit_event("comment_added", review, comment)
        return comment

    def reply_to_comment(
        self,
        review_id: str,
        comment_id: str,
        author_id: str,
        content: str,
    ) -> ReviewComment:
        """Reply to a comment."""
        review = self.storage.get_review(review_id)
        if not review:
            raise ValueError(f"Review {review_id} not found")

        parent_comment = None
        for comment in review.comments:
            if comment.id == comment_id:
                parent_comment = comment
                break

        if not parent_comment:
            raise ValueError(f"Comment {comment_id} not found")

        reply = ReviewComment(
            id=str(uuid4()),
            review_id=review_id,
            author_id=author_id,
            comment_type=CommentType.GENERAL,
            content=content,
        )

        parent_comment.replies.append(reply)
        parent_comment.updated_at = datetime.now()
        review.updated_at = datetime.now()
        self.storage.save_review(review)
        self._emit_event("reply_added", review, parent_comment, reply)
        return reply

    def resolve_comment(
        self,
        review_id: str,
        comment_id: str,
        resolver_id: str,
    ) -> ReviewComment:
        """Resolve a comment."""
        review = self.storage.get_review(review_id)
        if not review:
            raise ValueError(f"Review {review_id} not found")

        for comment in review.comments:
            if comment.id == comment_id:
                comment.resolved = True
                comment.resolved_by = resolver_id
                comment.resolved_at = datetime.now()
                comment.updated_at = datetime.now()
                review.updated_at = datetime.now()
                self.storage.save_review(review)
                self._emit_event("comment_resolved", review, comment)
                return comment

        raise ValueError(f"Comment {comment_id} not found")

    def add_suggestion(
        self,
        review_id: str,
        author_id: str,
        original_text: str,
        suggested_text: str,
        line_start: int,
        line_end: int,
        rationale: str,
    ) -> ReviewSuggestion:
        """Add a suggestion to a review."""
        review = self.storage.get_review(review_id)
        if not review:
            raise ValueError(f"Review {review_id} not found")

        suggestion = ReviewSuggestion(
            id=str(uuid4()),
            review_id=review_id,
            author_id=author_id,
            original_text=original_text,
            suggested_text=suggested_text,
            line_start=line_start,
            line_end=line_end,
            rationale=rationale,
        )

        review.suggestions.append(suggestion)
        review.updated_at = datetime.now()
        self.storage.save_review(review)
        self._emit_event("suggestion_added", review, suggestion)
        return suggestion

    def accept_suggestion(
        self,
        review_id: str,
        suggestion_id: str,
        accepter_id: str,
    ) -> ReviewSuggestion:
        """Accept a suggestion."""
        review = self.storage.get_review(review_id)
        if not review:
            raise ValueError(f"Review {review_id} not found")

        for suggestion in review.suggestions:
            if suggestion.id == suggestion_id:
                suggestion.accepted = True
                suggestion.accepted_by = accepter_id
                suggestion.accepted_at = datetime.now()
                review.updated_at = datetime.now()
                self.storage.save_review(review)
                self._emit_event("suggestion_accepted", review, suggestion)
                return suggestion

        raise ValueError(f"Suggestion {suggestion_id} not found")

    def reject_suggestion(
        self,
        review_id: str,
        suggestion_id: str,
        rejecter_id: str,
    ) -> ReviewSuggestion:
        """Reject a suggestion."""
        review = self.storage.get_review(review_id)
        if not review:
            raise ValueError(f"Review {review_id} not found")

        for suggestion in review.suggestions:
            if suggestion.id == suggestion_id:
                suggestion.accepted = False
                suggestion.accepted_by = rejecter_id
                suggestion.accepted_at = datetime.now()
                review.updated_at = datetime.now()
                self.storage.save_review(review)
                self._emit_event("suggestion_rejected", review, suggestion)
                return suggestion

        raise ValueError(f"Suggestion {suggestion_id} not found")

    def vote(
        self,
        review_id: str,
        reviewer_id: str,
        vote_type: VoteType,
        comment: Optional[str] = None,
    ) -> ReviewVote:
        """Cast a vote on a review."""
        review = self.storage.get_review(review_id)
        if not review:
            raise ValueError(f"Review {review_id} not found")

        if reviewer_id not in review.assigned_reviewers:
            raise ValueError(f"Reviewer {reviewer_id} is not assigned to this review")

        # Remove any existing vote from this reviewer
        review.votes = [v for v in review.votes if v.reviewer_id != reviewer_id]

        vote = ReviewVote(
            id=str(uuid4()),
            review_id=review_id,
            reviewer_id=reviewer_id,
            vote_type=vote_type,
            comment=comment,
        )

        review.votes.append(vote)

        # Update review status based on votes
        if vote_type == VoteType.REQUEST_CHANGES:
            review.status = ReviewStatus.CHANGES_REQUESTED
        elif vote_type == VoteType.REJECT:
            review.status = ReviewStatus.REJECTED
        elif review.is_approved() and not review.has_blocking_comments():
            review.status = ReviewStatus.APPROVED

        review.updated_at = datetime.now()
        self.storage.save_review(review)

        # Update reviewer metrics
        reviewer = self.storage.get_reviewer(reviewer_id)
        if reviewer:
            reviewer.reviews_completed += 1
            self.storage.save_reviewer(reviewer)

        self._emit_event("vote_cast", review, vote)
        return vote

    def merge_review(self, review_id: str, merger_id: str) -> ReviewRequest:
        """Merge an approved review."""
        review = self.storage.get_review(review_id)
        if not review:
            raise ValueError(f"Review {review_id} not found")

        if review.status != ReviewStatus.APPROVED:
            raise ValueError(f"Review is not approved: {review.status.name}")

        review.status = ReviewStatus.MERGED
        review.merged_at = datetime.now()
        review.updated_at = datetime.now()
        self.storage.save_review(review)
        self._emit_event("review_merged", review, merger_id)
        return review

    def close_review(
        self,
        review_id: str,
        closer_id: str,
        reason: Optional[str] = None,
    ) -> ReviewRequest:
        """Close a review without merging."""
        review = self.storage.get_review(review_id)
        if not review:
            raise ValueError(f"Review {review_id} not found")

        review.status = ReviewStatus.CLOSED
        review.closed_at = datetime.now()
        review.updated_at = datetime.now()
        self.storage.save_review(review)
        self._emit_event("review_closed", review, closer_id, reason)
        return review

    def get_review(self, review_id: str) -> Optional[ReviewRequest]:
        """Get a review by ID."""
        return self.storage.get_review(review_id)

    def list_reviews(
        self,
        status: Optional[ReviewStatus] = None,
        author_id: Optional[str] = None,
        reviewer_id: Optional[str] = None,
    ) -> List[ReviewRequest]:
        """List reviews with optional filters."""
        return self.storage.list_reviews(status, author_id, reviewer_id)

    def get_reviewer_metrics(self, reviewer_id: str) -> ReviewMetrics:
        """Get metrics for a reviewer."""
        reviewer = self.storage.get_reviewer(reviewer_id)
        if not reviewer:
            return ReviewMetrics()

        reviews = self.storage.list_reviews(reviewer_id=reviewer_id)

        total = len(reviews)
        completed = sum(
            1 for r in reviews if r.status in (ReviewStatus.MERGED, ReviewStatus.CLOSED)
        )

        # Calculate metrics
        total_comments = sum(
            len([c for c in r.comments if c.author_id == reviewer_id]) for r in reviews
        )
        avg_comments = total_comments / total if total > 0 else 0

        approvals = sum(
            1
            for r in reviews
            for v in r.votes
            if v.reviewer_id == reviewer_id and v.vote_type == VoteType.APPROVE
        )
        approval_rate = approvals / total if total > 0 else 0

        return ReviewMetrics(
            total_reviews=total,
            reviews_completed=completed,
            average_response_time_hours=reviewer.average_response_time_hours,
            average_comments_per_review=avg_comments,
            approval_rate=approval_rate,
        )

    def on(self, event: str, handler: Callable) -> None:
        """Register an event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def off(self, event: str, handler: Callable) -> None:
        """Unregister an event handler."""
        if event in self._event_handlers:
            self._event_handlers[event] = [h for h in self._event_handlers[event] if h != handler]

    def _emit_event(self, event: str, *args, **kwargs) -> None:
        """Emit an event to all registered handlers."""
        if event in self._event_handlers:
            for handler in self._event_handlers[event]:
                try:
                    handler(*args, **kwargs)
                except Exception:
                    pass  # Don't let handler errors affect the main flow


# Global reviewer instance
_global_reviewer: Optional[ExplanationReviewer] = None


def get_reviewer() -> ExplanationReviewer:
    """Get the global reviewer instance."""
    global _global_reviewer
    if _global_reviewer is None:
        _global_reviewer = ExplanationReviewer()
    return _global_reviewer


def reset_reviewer() -> None:
    """Reset the global reviewer instance."""
    global _global_reviewer
    _global_reviewer = None


def create_reviewer(
    storage: Optional[ReviewStorage] = None,
    matcher: Optional[ReviewerMatcher] = None,
) -> ExplanationReviewer:
    """Create a new reviewer instance."""
    return ExplanationReviewer(storage=storage, matcher=matcher)


# Convenience functions
def create_review_request(
    title: str,
    description: str,
    explanation_id: str,
    explanation_content: str,
    code_reference: str,
    author_id: str,
    **kwargs,
) -> ReviewRequest:
    """Create a new review request using the global reviewer."""
    reviewer = get_reviewer()
    return reviewer.create_review(
        title=title,
        description=description,
        explanation_id=explanation_id,
        explanation_content=explanation_content,
        code_reference=code_reference,
        author_id=author_id,
        **kwargs,
    )


def submit_for_review(review_id: str) -> ReviewRequest:
    """Submit a review using the global reviewer."""
    return get_reviewer().submit_review(review_id)


def add_review_comment(
    review_id: str,
    author_id: str,
    content: str,
    comment_type: CommentType = CommentType.GENERAL,
    **kwargs,
) -> ReviewComment:
    """Add a comment using the global reviewer."""
    return get_reviewer().add_comment(
        review_id=review_id,
        author_id=author_id,
        content=content,
        comment_type=comment_type,
        **kwargs,
    )


def add_review_suggestion(
    review_id: str,
    author_id: str,
    original_text: str,
    suggested_text: str,
    line_start: int,
    line_end: int,
    rationale: str,
) -> ReviewSuggestion:
    """Add a suggestion using the global reviewer."""
    return get_reviewer().add_suggestion(
        review_id=review_id,
        author_id=author_id,
        original_text=original_text,
        suggested_text=suggested_text,
        line_start=line_start,
        line_end=line_end,
        rationale=rationale,
    )


def approve_review(
    review_id: str,
    reviewer_id: str,
    comment: Optional[str] = None,
) -> ReviewVote:
    """Approve a review using the global reviewer."""
    return get_reviewer().vote(
        review_id=review_id,
        reviewer_id=reviewer_id,
        vote_type=VoteType.APPROVE,
        comment=comment,
    )


def request_changes(
    review_id: str,
    reviewer_id: str,
    comment: str,
) -> ReviewVote:
    """Request changes using the global reviewer."""
    return get_reviewer().vote(
        review_id=review_id,
        reviewer_id=reviewer_id,
        vote_type=VoteType.REQUEST_CHANGES,
        comment=comment,
    )
