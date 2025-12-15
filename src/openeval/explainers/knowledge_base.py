"""Knowledge base for storing and retrieving learned explanation patterns.

This module provides a comprehensive knowledge management system for code
explanations, enabling storage, retrieval, and recommendation of explanation
patterns based on historical data and similarity matching.

Example:
    >>> from openeval.explainers import KnowledgeBase, KnowledgeEntry
    >>> kb = get_knowledge_base()
    >>> entry = KnowledgeEntry(
    ...     pattern_id="async-pattern-001",
    ...     code_pattern="async def.*await",
    ...     explanation_template="Asynchronous function that...",
    ...     tags=["async", "concurrency"],
    ... )
    >>> kb.add_entry(entry)
    >>> recommendations = kb.get_recommendations("async def fetch_data():")
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class EntryType(Enum):
    """Types of knowledge entries."""

    PATTERN = "pattern"
    TEMPLATE = "template"
    EXAMPLE = "example"
    RULE = "rule"
    BEST_PRACTICE = "best_practice"


class ConfidenceLevel(Enum):
    """Confidence levels for knowledge entries."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"


class MatchType(Enum):
    """Types of pattern matching."""

    EXACT = "exact"
    REGEX = "regex"
    SEMANTIC = "semantic"
    FUZZY = "fuzzy"


@dataclass
class KnowledgeEntry:
    """A single entry in the knowledge base."""

    pattern_id: str
    code_pattern: str
    explanation_template: str
    entry_type: EntryType = EntryType.PATTERN
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    success_rate: float = 0.0
    examples: list[str] = field(default_factory=list)
    related_entries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary representation."""
        return {
            "pattern_id": self.pattern_id,
            "code_pattern": self.code_pattern,
            "explanation_template": self.explanation_template,
            "entry_type": self.entry_type.value,
            "confidence": self.confidence.value,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "examples": self.examples,
            "related_entries": self.related_entries,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeEntry:
        """Create entry from dictionary representation."""
        return cls(
            pattern_id=data["pattern_id"],
            code_pattern=data["code_pattern"],
            explanation_template=data["explanation_template"],
            entry_type=EntryType(data.get("entry_type", "pattern")),
            confidence=ConfidenceLevel(data.get("confidence", "medium")),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else datetime.now()
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if "updated_at" in data
                else datetime.now()
            ),
            usage_count=data.get("usage_count", 0),
            success_rate=data.get("success_rate", 0.0),
            examples=data.get("examples", []),
            related_entries=data.get("related_entries", []),
        )


@dataclass
class MatchResult:
    """Result of a pattern match."""

    entry: KnowledgeEntry
    score: float
    match_type: MatchType
    matched_portion: str
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence match."""
        return self.score >= 0.8 and self.entry.confidence in [
            ConfidenceLevel.HIGH,
            ConfidenceLevel.VERIFIED,
        ]


@dataclass
class Recommendation:
    """A recommendation from the knowledge base."""

    entry: KnowledgeEntry
    relevance_score: float
    reasoning: str
    suggested_template: str
    confidence: ConfidenceLevel
    alternatives: list[KnowledgeEntry] = field(default_factory=list)


@dataclass
class QueryContext:
    """Context for a knowledge base query."""

    code_snippet: str
    language: str = "python"
    file_path: str | None = None
    surrounding_context: str = ""
    user_preferences: dict[str, Any] = field(default_factory=dict)
    max_results: int = 5
    min_confidence: ConfidenceLevel = ConfidenceLevel.LOW


@dataclass
class LearningFeedback:
    """Feedback for learning from user interactions."""

    entry_id: str
    was_helpful: bool
    user_rating: int = 0  # 1-5 scale
    user_comment: str = ""
    actual_explanation: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class KnowledgeStorage(ABC):
    """Abstract base class for knowledge storage backends."""

    @abstractmethod
    def save(self, entry: KnowledgeEntry) -> None:
        """Save an entry to storage."""
        pass

    @abstractmethod
    def load(self, pattern_id: str) -> KnowledgeEntry | None:
        """Load an entry from storage."""
        pass

    @abstractmethod
    def delete(self, pattern_id: str) -> bool:
        """Delete an entry from storage."""
        pass

    @abstractmethod
    def list_all(self) -> list[KnowledgeEntry]:
        """List all entries in storage."""
        pass

    @abstractmethod
    def search(self, query: str, tags: list[str] | None = None) -> list[KnowledgeEntry]:
        """Search entries by query and tags."""
        pass


class InMemoryKnowledgeStorage(KnowledgeStorage):
    """In-memory storage for knowledge entries."""

    def __init__(self) -> None:
        """Initialize in-memory storage."""
        self._entries: dict[str, KnowledgeEntry] = {}
        self._tag_index: dict[str, set[str]] = {}

    def save(self, entry: KnowledgeEntry) -> None:
        """Save an entry to memory."""
        self._entries[entry.pattern_id] = entry
        for tag in entry.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(entry.pattern_id)

    def load(self, pattern_id: str) -> KnowledgeEntry | None:
        """Load an entry from memory."""
        return self._entries.get(pattern_id)

    def delete(self, pattern_id: str) -> bool:
        """Delete an entry from memory."""
        if pattern_id in self._entries:
            entry = self._entries.pop(pattern_id)
            for tag in entry.tags:
                if tag in self._tag_index:
                    self._tag_index[tag].discard(pattern_id)
            return True
        return False

    def list_all(self) -> list[KnowledgeEntry]:
        """List all entries."""
        return list(self._entries.values())

    def search(self, query: str, tags: list[str] | None = None) -> list[KnowledgeEntry]:
        """Search entries by query and tags."""
        results = []
        query_lower = query.lower()

        for entry in self._entries.values():
            # Check tag filter
            if tags:
                if not any(tag in entry.tags for tag in tags):
                    continue

            # Check query match
            if (
                query_lower in entry.code_pattern.lower()
                or query_lower in entry.explanation_template.lower()
                or any(query_lower in tag.lower() for tag in entry.tags)
            ):
                results.append(entry)

        return results


class FileKnowledgeStorage(KnowledgeStorage):
    """File-based storage for knowledge entries."""

    def __init__(self, storage_dir: Path | str) -> None:
        """Initialize file storage.

        Args:
            storage_dir: Directory to store knowledge files.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.storage_dir / "index.json"
        self._index: dict[str, str] = self._load_index()

    def _load_index(self) -> dict[str, str]:
        """Load the index file."""
        if self._index_file.exists():
            with open(self._index_file) as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        """Save the index file."""
        with open(self._index_file, "w") as f:
            json.dump(self._index, f, indent=2)

    def _get_file_path(self, pattern_id: str) -> Path:
        """Get file path for an entry."""
        safe_id = hashlib.md5(pattern_id.encode()).hexdigest()
        return self.storage_dir / f"{safe_id}.json"

    def save(self, entry: KnowledgeEntry) -> None:
        """Save an entry to file."""
        file_path = self._get_file_path(entry.pattern_id)
        with open(file_path, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)
        self._index[entry.pattern_id] = str(file_path)
        self._save_index()

    def load(self, pattern_id: str) -> KnowledgeEntry | None:
        """Load an entry from file."""
        if pattern_id not in self._index:
            return None
        file_path = Path(self._index[pattern_id])
        if not file_path.exists():
            return None
        with open(file_path) as f:
            data = json.load(f)
        return KnowledgeEntry.from_dict(data)

    def delete(self, pattern_id: str) -> bool:
        """Delete an entry from file."""
        if pattern_id not in self._index:
            return False
        file_path = Path(self._index.pop(pattern_id))
        self._save_index()
        if file_path.exists():
            file_path.unlink()
        return True

    def list_all(self) -> list[KnowledgeEntry]:
        """List all entries."""
        entries = []
        for pattern_id in self._index:
            entry = self.load(pattern_id)
            if entry:
                entries.append(entry)
        return entries

    def search(self, query: str, tags: list[str] | None = None) -> list[KnowledgeEntry]:
        """Search entries by query and tags."""
        results = []
        query_lower = query.lower()

        for entry in self.list_all():
            if tags and not any(tag in entry.tags for tag in tags):
                continue

            if (
                query_lower in entry.code_pattern.lower()
                or query_lower in entry.explanation_template.lower()
                or any(query_lower in tag.lower() for tag in entry.tags)
            ):
                results.append(entry)

        return results


class PatternMatcher:
    """Matches code against knowledge patterns."""

    def __init__(self) -> None:
        """Initialize the pattern matcher."""
        self._compiled_patterns: dict[str, re.Pattern] = {}
        self._custom_matchers: dict[str, Callable[[str, str], float]] = {}

    def register_matcher(self, name: str, matcher: Callable[[str, str], float]) -> None:
        """Register a custom matcher function.

        Args:
            name: Name of the matcher.
            matcher: Function that takes (code, pattern) and returns a score.
        """
        self._custom_matchers[name] = matcher

    def match_exact(self, code: str, pattern: str) -> float:
        """Exact string matching."""
        if pattern in code:
            return 1.0
        return 0.0

    def match_regex(self, code: str, pattern: str) -> float:
        """Regex pattern matching."""
        try:
            if pattern not in self._compiled_patterns:
                self._compiled_patterns[pattern] = re.compile(pattern, re.MULTILINE | re.DOTALL)
            compiled = self._compiled_patterns[pattern]
            matches = compiled.findall(code)
            if matches:
                # Score based on coverage
                total_match_len = sum(len(m) if isinstance(m, str) else len(m[0]) for m in matches)
                return min(1.0, total_match_len / len(code) * 2)
        except re.error:
            pass
        return 0.0

    def match_fuzzy(self, code: str, pattern: str) -> float:
        """Fuzzy string matching using simple similarity."""
        # Simple Jaccard-like similarity on tokens
        code_tokens = set(re.findall(r"\w+", code.lower()))
        pattern_tokens = set(re.findall(r"\w+", pattern.lower()))

        if not pattern_tokens:
            return 0.0

        intersection = code_tokens & pattern_tokens
        union = code_tokens | pattern_tokens

        return len(intersection) / len(union) if union else 0.0

    def match_semantic(self, code: str, pattern: str) -> float:
        """Semantic matching based on code structure.

        This is a simplified version - could be enhanced with embeddings.
        """
        # Extract code features
        code_features = self._extract_features(code)
        pattern_features = self._extract_features(pattern)

        if not pattern_features:
            return 0.0

        # Calculate overlap
        matches = sum(1 for f in pattern_features if f in code_features)
        return matches / len(pattern_features)

    def _extract_features(self, text: str) -> set[str]:
        """Extract code features for semantic matching."""
        features = set()

        # Keywords
        keywords = {
            "async",
            "await",
            "def",
            "class",
            "if",
            "for",
            "while",
            "try",
            "except",
            "with",
            "yield",
            "return",
            "import",
            "from",
            "lambda",
            "raise",
        }
        for keyword in keywords:
            if re.search(rf"\b{keyword}\b", text):
                features.add(f"keyword:{keyword}")

        # Patterns
        if re.search(r"async\s+def", text):
            features.add("pattern:async_function")
        if re.search(r"@\w+", text):
            features.add("pattern:decorator")
        if re.search(r"class\s+\w+.*:", text):
            features.add("pattern:class_definition")
        if re.search(r"def\s+__\w+__", text):
            features.add("pattern:dunder_method")
        if re.search(r"try:.*except", text, re.DOTALL):
            features.add("pattern:exception_handling")
        if re.search(r"with\s+\w+.*:", text):
            features.add("pattern:context_manager")
        if re.search(r"yield\s+", text):
            features.add("pattern:generator")
        if re.search(r"\[.*for.*in.*\]", text):
            features.add("pattern:list_comprehension")

        return features

    def match(
        self, code: str, entry: KnowledgeEntry, match_type: MatchType | None = None
    ) -> MatchResult | None:
        """Match code against a knowledge entry.

        Args:
            code: The code to match.
            entry: The knowledge entry to match against.
            match_type: Specific match type to use, or None to try all.

        Returns:
            MatchResult if match found, None otherwise.
        """
        best_score = 0.0
        best_type = MatchType.EXACT
        matched_portion = ""

        match_methods = {
            MatchType.EXACT: self.match_exact,
            MatchType.REGEX: self.match_regex,
            MatchType.FUZZY: self.match_fuzzy,
            MatchType.SEMANTIC: self.match_semantic,
        }

        types_to_try = [match_type] if match_type else list(MatchType)

        for mt in types_to_try:
            if mt in match_methods:
                score = match_methods[mt](code, entry.code_pattern)
                if score > best_score:
                    best_score = score
                    best_type = mt
                    # Extract matched portion for regex
                    if mt == MatchType.REGEX:
                        try:
                            match = re.search(entry.code_pattern, code)
                            if match:
                                matched_portion = match.group(0)
                        except re.error:
                            matched_portion = ""
                    else:
                        matched_portion = entry.code_pattern if score > 0 else ""

        if best_score > 0:
            return MatchResult(
                entry=entry,
                score=best_score,
                match_type=best_type,
                matched_portion=matched_portion,
            )
        return None


class RecommendationEngine:
    """Engine for generating recommendations from the knowledge base."""

    def __init__(self, storage: KnowledgeStorage) -> None:
        """Initialize the recommendation engine.

        Args:
            storage: Storage backend for knowledge entries.
        """
        self.storage = storage
        self.matcher = PatternMatcher()
        self._feature_weights = {
            "match_score": 0.4,
            "confidence": 0.2,
            "usage_count": 0.2,
            "success_rate": 0.2,
        }

    def get_recommendations(self, context: QueryContext) -> list[Recommendation]:
        """Get recommendations for given code context.

        Args:
            context: Query context including code and preferences.

        Returns:
            List of recommendations sorted by relevance.
        """
        all_entries = self.storage.list_all()
        recommendations = []

        for entry in all_entries:
            # Check confidence threshold
            confidence_order = [
                ConfidenceLevel.LOW,
                ConfidenceLevel.MEDIUM,
                ConfidenceLevel.HIGH,
                ConfidenceLevel.VERIFIED,
            ]
            if confidence_order.index(entry.confidence) < confidence_order.index(
                context.min_confidence
            ):
                continue

            # Match against the code
            match_result = self.matcher.match(context.code_snippet, entry)
            if not match_result or match_result.score < 0.1:
                continue

            # Calculate relevance score
            relevance = self._calculate_relevance(entry, match_result, context)

            # Generate reasoning
            reasoning = self._generate_reasoning(entry, match_result, context)

            # Create suggestion
            suggested_template = self._customize_template(entry.explanation_template, context)

            recommendation = Recommendation(
                entry=entry,
                relevance_score=relevance,
                reasoning=reasoning,
                suggested_template=suggested_template,
                confidence=entry.confidence,
            )
            recommendations.append(recommendation)

        # Sort by relevance
        recommendations.sort(key=lambda r: r.relevance_score, reverse=True)

        # Limit results
        top_recommendations = recommendations[: context.max_results]

        # Add alternatives
        for rec in top_recommendations:
            rec.alternatives = [
                r.entry for r in recommendations if r.entry.pattern_id != rec.entry.pattern_id
            ][:3]

        return top_recommendations

    def _calculate_relevance(
        self,
        entry: KnowledgeEntry,
        match_result: MatchResult,
        context: QueryContext,
    ) -> float:
        """Calculate relevance score for an entry."""
        scores = {
            "match_score": match_result.score,
            "confidence": {
                ConfidenceLevel.LOW: 0.25,
                ConfidenceLevel.MEDIUM: 0.5,
                ConfidenceLevel.HIGH: 0.75,
                ConfidenceLevel.VERIFIED: 1.0,
            }[entry.confidence],
            "usage_count": min(1.0, entry.usage_count / 100),  # Normalize
            "success_rate": entry.success_rate,
        }

        # Apply tag bonus
        if context.user_preferences.get("preferred_tags"):
            tag_match = len(set(entry.tags) & set(context.user_preferences["preferred_tags"]))
            scores["match_score"] *= 1 + (tag_match * 0.1)

        # Weighted average
        total = sum(scores[key] * self._feature_weights[key] for key in self._feature_weights)
        return min(1.0, total)

    def _generate_reasoning(
        self,
        entry: KnowledgeEntry,
        match_result: MatchResult,
        context: QueryContext,
    ) -> str:
        """Generate reasoning for a recommendation."""
        reasons = []

        if match_result.score >= 0.8:
            reasons.append(f"Strong pattern match ({match_result.match_type.value})")
        elif match_result.score >= 0.5:
            reasons.append(f"Moderate pattern match ({match_result.match_type.value})")
        else:
            reasons.append(f"Partial pattern match ({match_result.match_type.value})")

        if entry.confidence == ConfidenceLevel.VERIFIED:
            reasons.append("This is a verified explanation pattern")
        elif entry.confidence == ConfidenceLevel.HIGH:
            reasons.append("High confidence based on usage history")

        if entry.usage_count > 50:
            reasons.append(f"Used {entry.usage_count} times successfully")

        if entry.success_rate > 0.8:
            reasons.append(f"{entry.success_rate:.0%} user satisfaction rate")

        return ". ".join(reasons)

    def _customize_template(self, template: str, context: QueryContext) -> str:
        """Customize template based on context."""
        # Replace placeholders
        customized = template

        # Language-specific customization
        if context.language:
            customized = customized.replace("{{language}}", context.language)

        # Add context-aware modifications
        if context.file_path:
            customized = customized.replace("{{file}}", Path(context.file_path).name)

        return customized


class LearningEngine:
    """Engine for learning from user feedback."""

    def __init__(self, storage: KnowledgeStorage) -> None:
        """Initialize the learning engine.

        Args:
            storage: Storage backend for knowledge entries.
        """
        self.storage = storage
        self._feedback_history: list[LearningFeedback] = []
        self._learning_rate = 0.1

    def record_feedback(self, feedback: LearningFeedback) -> None:
        """Record user feedback for learning.

        Args:
            feedback: The feedback to record.
        """
        self._feedback_history.append(feedback)

        # Update entry based on feedback
        entry = self.storage.load(feedback.entry_id)
        if entry:
            entry.usage_count += 1
            # Update success rate using exponential moving average
            current_success = 1.0 if feedback.was_helpful else 0.0
            entry.success_rate = (
                self._learning_rate * current_success
                + (1 - self._learning_rate) * entry.success_rate
            )
            entry.updated_at = datetime.now()
            self.storage.save(entry)

    def learn_new_pattern(
        self,
        code: str,
        explanation: str,
        tags: list[str] | None = None,
        entry_type: EntryType = EntryType.PATTERN,
    ) -> KnowledgeEntry:
        """Learn a new pattern from a code-explanation pair.

        Args:
            code: The code snippet.
            explanation: The explanation for the code.
            tags: Optional tags for the pattern.
            entry_type: Type of the entry.

        Returns:
            The newly created knowledge entry.
        """
        # Generate pattern ID
        pattern_id = hashlib.md5(f"{code}:{explanation}:{time.time()}".encode()).hexdigest()[:16]

        # Auto-extract tags if not provided
        if tags is None:
            tags = self._extract_tags(code)

        entry = KnowledgeEntry(
            pattern_id=pattern_id,
            code_pattern=self._generalize_pattern(code),
            explanation_template=explanation,
            entry_type=entry_type,
            confidence=ConfidenceLevel.LOW,  # New patterns start low
            tags=tags,
            examples=[code],
        )

        self.storage.save(entry)
        return entry

    def _generalize_pattern(self, code: str) -> str:
        """Generalize code into a reusable pattern."""
        generalized = code

        # Replace specific names with wildcards for regex
        # Variable names
        generalized = re.sub(r"\b([a-z_][a-z0-9_]*)\b(?!\s*=)", r"\\w+", generalized)

        # String literals
        generalized = re.sub(r'"[^"]*"', r'"[^"]*"', generalized)
        generalized = re.sub(r"'[^']*'", r"'[^']*'", generalized)

        return generalized

    def _extract_tags(self, code: str) -> list[str]:
        """Automatically extract tags from code."""
        tags = []

        # Detect patterns and add tags
        if re.search(r"async\s+def", code):
            tags.append("async")
        if re.search(r"await\s+", code):
            tags.append("await")
        if re.search(r"class\s+\w+", code):
            tags.append("class")
        if re.search(r"@\w+", code):
            tags.append("decorator")
        if re.search(r"def\s+__\w+__", code):
            tags.append("dunder")
        if re.search(r"try:.*except", code, re.DOTALL):
            tags.append("exception-handling")
        if re.search(r"with\s+", code):
            tags.append("context-manager")
        if re.search(r"yield\s+", code):
            tags.append("generator")
        if re.search(r"lambda\s+", code):
            tags.append("lambda")
        if re.search(r"import\s+", code):
            tags.append("import")

        return tags

    def promote_entry(self, pattern_id: str) -> bool:
        """Promote an entry to higher confidence based on usage.

        Args:
            pattern_id: The ID of the entry to promote.

        Returns:
            True if entry was promoted, False otherwise.
        """
        entry = self.storage.load(pattern_id)
        if not entry:
            return False

        # Promotion criteria
        if (
            entry.confidence == ConfidenceLevel.LOW
            and entry.usage_count >= 10
            and entry.success_rate >= 0.6
        ):
            entry.confidence = ConfidenceLevel.MEDIUM
        elif (
            entry.confidence == ConfidenceLevel.MEDIUM
            and entry.usage_count >= 50
            and entry.success_rate >= 0.8
        ):
            entry.confidence = ConfidenceLevel.HIGH
        else:
            return False

        entry.updated_at = datetime.now()
        self.storage.save(entry)
        return True


class KnowledgeBase:
    """Main knowledge base class combining storage, matching, and learning."""

    def __init__(
        self,
        storage: KnowledgeStorage | None = None,
    ) -> None:
        """Initialize the knowledge base.

        Args:
            storage: Storage backend. Defaults to in-memory storage.
        """
        self.storage = storage or InMemoryKnowledgeStorage()
        self.matcher = PatternMatcher()
        self.recommendation_engine = RecommendationEngine(self.storage)
        self.learning_engine = LearningEngine(self.storage)
        self._initialized = True

    def add_entry(self, entry: KnowledgeEntry) -> None:
        """Add an entry to the knowledge base.

        Args:
            entry: The entry to add.
        """
        self.storage.save(entry)

    def get_entry(self, pattern_id: str) -> KnowledgeEntry | None:
        """Get an entry by ID.

        Args:
            pattern_id: The ID of the entry.

        Returns:
            The entry if found, None otherwise.
        """
        return self.storage.load(pattern_id)

    def remove_entry(self, pattern_id: str) -> bool:
        """Remove an entry from the knowledge base.

        Args:
            pattern_id: The ID of the entry to remove.

        Returns:
            True if entry was removed, False otherwise.
        """
        return self.storage.delete(pattern_id)

    def search(self, query: str, tags: list[str] | None = None) -> list[KnowledgeEntry]:
        """Search for entries.

        Args:
            query: Search query.
            tags: Optional tag filters.

        Returns:
            List of matching entries.
        """
        return self.storage.search(query, tags)

    def get_recommendations(
        self,
        code: str,
        language: str = "python",
        max_results: int = 5,
        **kwargs: Any,
    ) -> list[Recommendation]:
        """Get recommendations for code.

        Args:
            code: The code to get recommendations for.
            language: Programming language.
            max_results: Maximum number of recommendations.
            **kwargs: Additional context parameters.

        Returns:
            List of recommendations.
        """
        context = QueryContext(
            code_snippet=code,
            language=language,
            max_results=max_results,
            user_preferences=kwargs,
        )
        return self.recommendation_engine.get_recommendations(context)

    def learn(
        self,
        code: str,
        explanation: str,
        tags: list[str] | None = None,
    ) -> KnowledgeEntry:
        """Learn a new pattern from code and explanation.

        Args:
            code: The code snippet.
            explanation: The explanation.
            tags: Optional tags.

        Returns:
            The created entry.
        """
        return self.learning_engine.learn_new_pattern(code, explanation, tags)

    def record_feedback(
        self,
        entry_id: str,
        was_helpful: bool,
        rating: int = 0,
        comment: str = "",
    ) -> None:
        """Record feedback for an entry.

        Args:
            entry_id: The ID of the entry.
            was_helpful: Whether the entry was helpful.
            rating: User rating (1-5).
            comment: Optional comment.
        """
        feedback = LearningFeedback(
            entry_id=entry_id,
            was_helpful=was_helpful,
            user_rating=rating,
            user_comment=comment,
        )
        self.learning_engine.record_feedback(feedback)

    def get_statistics(self) -> dict[str, Any]:
        """Get knowledge base statistics.

        Returns:
            Dictionary of statistics.
        """
        entries = self.storage.list_all()
        total = len(entries)

        if total == 0:
            return {
                "total_entries": 0,
                "by_type": {},
                "by_confidence": {},
                "avg_usage_count": 0,
                "avg_success_rate": 0,
            }

        by_type: dict[str, int] = {}
        by_confidence: dict[str, int] = {}
        total_usage = 0
        total_success = 0.0

        for entry in entries:
            by_type[entry.entry_type.value] = by_type.get(entry.entry_type.value, 0) + 1
            by_confidence[entry.confidence.value] = by_confidence.get(entry.confidence.value, 0) + 1
            total_usage += entry.usage_count
            total_success += entry.success_rate

        return {
            "total_entries": total,
            "by_type": by_type,
            "by_confidence": by_confidence,
            "avg_usage_count": total_usage / total,
            "avg_success_rate": total_success / total,
        }

    def export_all(self) -> list[dict[str, Any]]:
        """Export all entries as dictionaries.

        Returns:
            List of entry dictionaries.
        """
        return [entry.to_dict() for entry in self.storage.list_all()]

    def import_entries(self, entries: list[dict[str, Any]]) -> int:
        """Import entries from dictionaries.

        Args:
            entries: List of entry dictionaries.

        Returns:
            Number of entries imported.
        """
        count = 0
        for data in entries:
            try:
                entry = KnowledgeEntry.from_dict(data)
                self.storage.save(entry)
                count += 1
            except (KeyError, ValueError):
                continue
        return count


# Global instance
_knowledge_base: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    """Get the global knowledge base instance.

    Returns:
        The global KnowledgeBase instance.
    """
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
    return _knowledge_base


def reset_knowledge_base() -> None:
    """Reset the global knowledge base instance."""
    global _knowledge_base
    _knowledge_base = None


def create_knowledge_base(
    storage_path: Path | str | None = None,
) -> KnowledgeBase:
    """Create a new knowledge base with optional file storage.

    Args:
        storage_path: Path for file-based storage, or None for in-memory.

    Returns:
        A new KnowledgeBase instance.
    """
    if storage_path:
        storage = FileKnowledgeStorage(storage_path)
    else:
        storage = InMemoryKnowledgeStorage()
    return KnowledgeBase(storage)


# Convenience functions
def add_knowledge_entry(
    pattern_id: str,
    code_pattern: str,
    explanation_template: str,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> KnowledgeEntry:
    """Add a knowledge entry to the global knowledge base.

    Args:
        pattern_id: Unique identifier for the pattern.
        code_pattern: The code pattern to match.
        explanation_template: Template for explanations.
        tags: Optional tags for the entry.
        **kwargs: Additional entry attributes.

    Returns:
        The created entry.
    """
    entry = KnowledgeEntry(
        pattern_id=pattern_id,
        code_pattern=code_pattern,
        explanation_template=explanation_template,
        tags=tags or [],
        **kwargs,
    )
    get_knowledge_base().add_entry(entry)
    return entry


def get_explanation_recommendations(
    code: str,
    max_results: int = 5,
) -> list[Recommendation]:
    """Get explanation recommendations for code.

    Args:
        code: The code to get recommendations for.
        max_results: Maximum number of recommendations.

    Returns:
        List of recommendations.
    """
    return get_knowledge_base().get_recommendations(code, max_results=max_results)


def learn_from_explanation(
    code: str,
    explanation: str,
    tags: list[str] | None = None,
) -> KnowledgeEntry:
    """Learn a new pattern from a code-explanation pair.

    Args:
        code: The code snippet.
        explanation: The explanation.
        tags: Optional tags.

    Returns:
        The created knowledge entry.
    """
    return get_knowledge_base().learn(code, explanation, tags)
