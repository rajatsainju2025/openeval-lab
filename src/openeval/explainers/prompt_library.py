"""Prompt library for managing and organizing explanation prompts.

This module provides a centralized library for prompts with categorization,
versioning, and A/B testing support.
"""

import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# Enums and Type Definitions
# =============================================================================


class PromptCategory(str, Enum):
    """Categories of prompts."""

    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    ALGORITHM = "algorithm"
    DATA_STRUCTURE = "data_structure"
    API = "api"
    ERROR_HANDLING = "error_handling"
    PERFORMANCE = "performance"
    SECURITY = "security"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    GENERAL = "general"


class PromptTone(str, Enum):
    """Tone of the prompt."""

    TECHNICAL = "technical"
    EDUCATIONAL = "educational"
    CONCISE = "concise"
    DETAILED = "detailed"
    FRIENDLY = "friendly"
    FORMAL = "formal"


class PromptStatus(str, Enum):
    """Status of a prompt."""

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class VariantSelectionStrategy(str, Enum):
    """Strategy for selecting prompt variants."""

    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    DETERMINISTIC = "deterministic"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PromptTemplate:
    """A prompt template with metadata."""

    id: str
    name: str
    template: str
    category: PromptCategory
    version: str = "1.0.0"
    tone: PromptTone = PromptTone.TECHNICAL
    status: PromptStatus = PromptStatus.ACTIVE
    description: str = ""
    variables: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    author: str = "system"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """Get hash of the template content."""
        return hashlib.sha256(self.template.encode()).hexdigest()[:12]

    def render(self, **kwargs: Any) -> str:
        """Render the template with variables.

        Args:
            **kwargs: Variables to substitute.

        Returns:
            Rendered prompt string.
        """
        result = self.template
        for key, value in kwargs.items():
            placeholder = f"{{{key}}}"
            result = result.replace(placeholder, str(value))
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "template": self.template,
            "category": self.category.value,
            "version": self.version,
            "tone": self.tone.value,
            "status": self.status.value,
            "description": self.description,
            "variables": self.variables,
            "tags": self.tags,
            "author": self.author,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            template=data["template"],
            category=PromptCategory(data["category"]),
            version=data.get("version", "1.0.0"),
            tone=PromptTone(data.get("tone", "technical")),
            status=PromptStatus(data.get("status", "active")),
            description=data.get("description", ""),
            variables=data.get("variables", []),
            tags=data.get("tags", []),
            author=data.get("author", "system"),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PromptVariant:
    """A variant of a prompt for A/B testing."""

    id: str
    prompt_id: str
    name: str
    template: str
    weight: float = 1.0
    impressions: int = 0
    conversions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def conversion_rate(self) -> float:
        """Calculate conversion rate."""
        if self.impressions == 0:
            return 0.0
        return self.conversions / self.impressions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "prompt_id": self.prompt_id,
            "name": self.name,
            "template": self.template,
            "weight": self.weight,
            "impressions": self.impressions,
            "conversions": self.conversions,
            "conversion_rate": self.conversion_rate,
        }


@dataclass
class ABTest:
    """An A/B test configuration."""

    id: str
    name: str
    prompt_id: str
    variants: List[PromptVariant] = field(default_factory=list)
    strategy: VariantSelectionStrategy = VariantSelectionStrategy.RANDOM
    is_active: bool = True
    start_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    end_date: Optional[str] = None
    winner_variant_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def select_variant(self, user_id: Optional[str] = None) -> PromptVariant:
        """Select a variant based on strategy.

        Args:
            user_id: Optional user ID for deterministic selection.

        Returns:
            Selected variant.
        """
        if not self.variants:
            raise ValueError("No variants available")

        if self.strategy == VariantSelectionStrategy.RANDOM:
            return random.choice(self.variants)

        elif self.strategy == VariantSelectionStrategy.WEIGHTED:
            total_weight = sum(v.weight for v in self.variants)
            r = random.uniform(0, total_weight)
            cumulative = 0
            for variant in self.variants:
                cumulative += variant.weight
                if r <= cumulative:
                    return variant
            return self.variants[-1]

        elif self.strategy == VariantSelectionStrategy.DETERMINISTIC:
            if user_id:
                index = hash(user_id) % len(self.variants)
                return self.variants[index]
            return self.variants[0]

        else:  # ROUND_ROBIN
            min_impressions = min(v.impressions for v in self.variants)
            for variant in self.variants:
                if variant.impressions == min_impressions:
                    return variant
            return self.variants[0]

    def record_impression(self, variant_id: str) -> None:
        """Record an impression for a variant."""
        for variant in self.variants:
            if variant.id == variant_id:
                variant.impressions += 1
                break

    def record_conversion(self, variant_id: str) -> None:
        """Record a conversion for a variant."""
        for variant in self.variants:
            if variant.id == variant_id:
                variant.conversions += 1
                break

    def get_results(self) -> Dict[str, Any]:
        """Get A/B test results."""
        return {
            "test_id": self.id,
            "name": self.name,
            "is_active": self.is_active,
            "variants": [v.to_dict() for v in self.variants],
            "winner": self.winner_variant_id,
            "total_impressions": sum(v.impressions for v in self.variants),
            "total_conversions": sum(v.conversions for v in self.variants),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "prompt_id": self.prompt_id,
            "variants": [v.to_dict() for v in self.variants],
            "strategy": self.strategy.value,
            "is_active": self.is_active,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "winner_variant_id": self.winner_variant_id,
        }


@dataclass
class PromptUsageStats:
    """Usage statistics for a prompt."""

    prompt_id: str
    total_uses: int = 0
    successful_uses: int = 0
    average_response_time: float = 0.0
    last_used: Optional[str] = None
    user_ratings: List[float] = field(default_factory=list)
    error_count: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_uses == 0:
            return 0.0
        return self.successful_uses / self.total_uses

    @property
    def average_rating(self) -> float:
        """Calculate average user rating."""
        if not self.user_ratings:
            return 0.0
        return sum(self.user_ratings) / len(self.user_ratings)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_id": self.prompt_id,
            "total_uses": self.total_uses,
            "successful_uses": self.successful_uses,
            "success_rate": self.success_rate,
            "average_response_time": self.average_response_time,
            "average_rating": self.average_rating,
            "last_used": self.last_used,
            "error_count": self.error_count,
        }


@dataclass
class LibraryConfig:
    """Configuration for the prompt library."""

    enable_versioning: bool = True
    enable_ab_testing: bool = True
    enable_usage_tracking: bool = True
    default_category: PromptCategory = PromptCategory.GENERAL
    default_tone: PromptTone = PromptTone.TECHNICAL
    max_prompt_length: int = 10000
    cache_prompts: bool = True


# =============================================================================
# Prompt Library
# =============================================================================


class PromptLibrary:
    """Central library for managing prompts."""

    def __init__(self, config: Optional[LibraryConfig] = None):
        """Initialize prompt library.

        Args:
            config: Optional configuration.
        """
        self._config = config or LibraryConfig()
        self._prompts: Dict[str, PromptTemplate] = {}
        self._versions: Dict[str, Dict[str, PromptTemplate]] = {}
        self._ab_tests: Dict[str, ABTest] = {}
        self._usage_stats: Dict[str, PromptUsageStats] = {}
        self._category_index: Dict[PromptCategory, Set[str]] = {
            cat: set() for cat in PromptCategory
        }
        self._tag_index: Dict[str, Set[str]] = {}

        # Load built-in prompts
        self._load_builtin_prompts()

    def add_prompt(self, prompt: PromptTemplate) -> None:
        """Add a prompt to the library.

        Args:
            prompt: Prompt template to add.
        """
        self._prompts[prompt.id] = prompt

        # Update indices
        self._category_index[prompt.category].add(prompt.id)
        for tag in prompt.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(prompt.id)

        # Track version
        if self._config.enable_versioning:
            if prompt.id not in self._versions:
                self._versions[prompt.id] = {}
            self._versions[prompt.id][prompt.version] = prompt

        # Initialize usage stats
        if self._config.enable_usage_tracking:
            if prompt.id not in self._usage_stats:
                self._usage_stats[prompt.id] = PromptUsageStats(prompt_id=prompt.id)

    def get_prompt(self, prompt_id: str, version: Optional[str] = None) -> Optional[PromptTemplate]:
        """Get a prompt by ID.

        Args:
            prompt_id: Prompt ID.
            version: Optional specific version.

        Returns:
            Prompt template if found.
        """
        if version and self._config.enable_versioning:
            versions = self._versions.get(prompt_id, {})
            return versions.get(version)
        return self._prompts.get(prompt_id)

    def search(
        self,
        query: Optional[str] = None,
        category: Optional[PromptCategory] = None,
        tags: Optional[List[str]] = None,
        tone: Optional[PromptTone] = None,
        status: Optional[PromptStatus] = None,
    ) -> List[PromptTemplate]:
        """Search for prompts.

        Args:
            query: Text search query.
            category: Filter by category.
            tags: Filter by tags.
            tone: Filter by tone.
            status: Filter by status.

        Returns:
            List of matching prompts.
        """
        results = list(self._prompts.values())

        if category:
            prompt_ids = self._category_index.get(category, set())
            results = [p for p in results if p.id in prompt_ids]

        if tags:
            for tag in tags:
                tag_ids = self._tag_index.get(tag, set())
                results = [p for p in results if p.id in tag_ids]

        if tone:
            results = [p for p in results if p.tone == tone]

        if status:
            results = [p for p in results if p.status == status]

        if query:
            query_lower = query.lower()
            results = [
                p
                for p in results
                if query_lower in p.name.lower()
                or query_lower in p.description.lower()
                or query_lower in p.template.lower()
            ]

        return results

    def get_by_category(self, category: PromptCategory) -> List[PromptTemplate]:
        """Get all prompts in a category.

        Args:
            category: Prompt category.

        Returns:
            List of prompts.
        """
        prompt_ids = self._category_index.get(category, set())
        return [self._prompts[pid] for pid in prompt_ids if pid in self._prompts]

    def get_by_tag(self, tag: str) -> List[PromptTemplate]:
        """Get all prompts with a tag.

        Args:
            tag: Tag to search for.

        Returns:
            List of prompts.
        """
        prompt_ids = self._tag_index.get(tag, set())
        return [self._prompts[pid] for pid in prompt_ids if pid in self._prompts]

    def update_prompt(
        self, prompt_id: str, updates: Dict[str, Any], new_version: bool = True
    ) -> Optional[PromptTemplate]:
        """Update a prompt.

        Args:
            prompt_id: ID of prompt to update.
            updates: Dictionary of updates.
            new_version: Whether to create a new version.

        Returns:
            Updated prompt if found.
        """
        prompt = self._prompts.get(prompt_id)
        if not prompt:
            return None

        # Create new version if enabled
        if new_version and self._config.enable_versioning:
            old_version_parts = prompt.version.split(".")
            new_version_num = f"{old_version_parts[0]}.{int(old_version_parts[1]) + 1}.0"
        else:
            new_version_num = prompt.version

        # Apply updates
        updated = PromptTemplate(
            id=prompt.id,
            name=updates.get("name", prompt.name),
            template=updates.get("template", prompt.template),
            category=updates.get("category", prompt.category),
            version=new_version_num,
            tone=updates.get("tone", prompt.tone),
            status=updates.get("status", prompt.status),
            description=updates.get("description", prompt.description),
            variables=updates.get("variables", prompt.variables),
            tags=updates.get("tags", prompt.tags),
            author=updates.get("author", prompt.author),
            created_at=prompt.created_at,
            updated_at=datetime.utcnow().isoformat(),
            metadata=updates.get("metadata", prompt.metadata),
        )

        self.add_prompt(updated)
        return updated

    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt.

        Args:
            prompt_id: ID of prompt to delete.

        Returns:
            True if deleted.
        """
        prompt = self._prompts.get(prompt_id)
        if not prompt:
            return False

        # Remove from indices
        self._category_index[prompt.category].discard(prompt_id)
        for tag in prompt.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(prompt_id)

        del self._prompts[prompt_id]
        return True

    # A/B Testing Methods

    def create_ab_test(
        self,
        name: str,
        prompt_id: str,
        variants: List[Dict[str, Any]],
        strategy: VariantSelectionStrategy = VariantSelectionStrategy.RANDOM,
    ) -> ABTest:
        """Create an A/B test.

        Args:
            name: Test name.
            prompt_id: Base prompt ID.
            variants: List of variant definitions.
            strategy: Selection strategy.

        Returns:
            Created A/B test.
        """
        test_id = f"test_{prompt_id}_{datetime.utcnow().timestamp()}"

        variant_objects = []
        for i, v in enumerate(variants):
            variant = PromptVariant(
                id=f"{test_id}_v{i}",
                prompt_id=prompt_id,
                name=v.get("name", f"Variant {i}"),
                template=v["template"],
                weight=v.get("weight", 1.0),
            )
            variant_objects.append(variant)

        test = ABTest(
            id=test_id,
            name=name,
            prompt_id=prompt_id,
            variants=variant_objects,
            strategy=strategy,
        )

        self._ab_tests[test_id] = test
        return test

    def get_ab_test(self, test_id: str) -> Optional[ABTest]:
        """Get an A/B test by ID."""
        return self._ab_tests.get(test_id)

    def get_active_tests(self) -> List[ABTest]:
        """Get all active A/B tests."""
        return [t for t in self._ab_tests.values() if t.is_active]

    def conclude_ab_test(
        self, test_id: str, winner_variant_id: Optional[str] = None
    ) -> Optional[ABTest]:
        """Conclude an A/B test.

        Args:
            test_id: Test ID.
            winner_variant_id: Optional winner ID (auto-detected if None).

        Returns:
            Updated test.
        """
        test = self._ab_tests.get(test_id)
        if not test:
            return None

        test.is_active = False
        test.end_date = datetime.utcnow().isoformat()

        if winner_variant_id:
            test.winner_variant_id = winner_variant_id
        else:
            # Auto-detect winner by conversion rate
            if test.variants:
                best = max(test.variants, key=lambda v: v.conversion_rate)
                test.winner_variant_id = best.id

        return test

    # Usage Tracking Methods

    def record_usage(
        self,
        prompt_id: str,
        success: bool = True,
        response_time: float = 0.0,
        rating: Optional[float] = None,
    ) -> None:
        """Record prompt usage.

        Args:
            prompt_id: Prompt ID.
            success: Whether use was successful.
            response_time: Response time in seconds.
            rating: Optional user rating (0-5).
        """
        if not self._config.enable_usage_tracking:
            return

        if prompt_id not in self._usage_stats:
            self._usage_stats[prompt_id] = PromptUsageStats(prompt_id=prompt_id)

        stats = self._usage_stats[prompt_id]
        stats.total_uses += 1
        if success:
            stats.successful_uses += 1
        else:
            stats.error_count += 1

        # Update average response time
        if response_time > 0:
            total_time = stats.average_response_time * (stats.total_uses - 1)
            stats.average_response_time = (total_time + response_time) / stats.total_uses

        if rating is not None:
            stats.user_ratings.append(rating)

        stats.last_used = datetime.utcnow().isoformat()

    def get_usage_stats(self, prompt_id: str) -> Optional[PromptUsageStats]:
        """Get usage statistics for a prompt."""
        return self._usage_stats.get(prompt_id)

    def get_top_prompts(self, limit: int = 10) -> List[Tuple[PromptTemplate, PromptUsageStats]]:
        """Get top prompts by usage.

        Args:
            limit: Maximum number of results.

        Returns:
            List of (prompt, stats) tuples.
        """
        sorted_stats = sorted(
            self._usage_stats.values(),
            key=lambda s: s.total_uses,
            reverse=True,
        )[:limit]

        results = []
        for stats in sorted_stats:
            prompt = self._prompts.get(stats.prompt_id)
            if prompt:
                results.append((prompt, stats))

        return results

    def get_library_stats(self) -> Dict[str, Any]:
        """Get overall library statistics."""
        total_prompts = len(self._prompts)
        category_counts = {cat.value: len(ids) for cat, ids in self._category_index.items()}
        active_tests = len([t for t in self._ab_tests.values() if t.is_active])

        return {
            "total_prompts": total_prompts,
            "category_distribution": category_counts,
            "total_tags": len(self._tag_index),
            "active_ab_tests": active_tests,
            "total_ab_tests": len(self._ab_tests),
        }

    def _load_builtin_prompts(self) -> None:
        """Load built-in prompts."""
        builtins = [
            PromptTemplate(
                id="explain_function",
                name="Explain Function",
                template="""Analyze and explain this {language} function:

```{language}
{code}
```

Provide:
1. Purpose: What does this function do?
2. Parameters: Explain each parameter
3. Return value: What does it return?
4. Algorithm: How does it work step by step?
5. Edge cases: What edge cases does it handle?""",
                category=PromptCategory.FUNCTION,
                variables=["language", "code"],
                tags=["function", "explanation", "detailed"],
            ),
            PromptTemplate(
                id="explain_class",
                name="Explain Class",
                template="""Analyze and explain this {language} class:

```{language}
{code}
```

Provide:
1. Purpose: What does this class represent?
2. Attributes: Explain key attributes
3. Methods: Explain public methods
4. Design patterns: Any patterns used?
5. Usage: How to use this class?""",
                category=PromptCategory.CLASS,
                variables=["language", "code"],
                tags=["class", "explanation", "detailed"],
            ),
            PromptTemplate(
                id="explain_algorithm",
                name="Explain Algorithm",
                template="""Analyze this algorithm:

```{language}
{code}
```

Explain:
1. Algorithm type and purpose
2. Time complexity: O(?)
3. Space complexity: O(?)
4. Step-by-step walkthrough
5. Potential optimizations""",
                category=PromptCategory.ALGORITHM,
                variables=["language", "code"],
                tags=["algorithm", "complexity", "optimization"],
            ),
            PromptTemplate(
                id="explain_concise",
                name="Concise Explanation",
                template="Briefly explain what this code does in 2-3 sentences:\n\n```{language}\n{code}\n```",
                category=PromptCategory.GENERAL,
                tone=PromptTone.CONCISE,
                variables=["language", "code"],
                tags=["concise", "quick"],
            ),
            PromptTemplate(
                id="explain_beginner",
                name="Beginner-Friendly Explanation",
                template="""Explain this code to someone learning to program:

```{language}
{code}
```

Use simple language and analogies. Explain any technical terms.""",
                category=PromptCategory.GENERAL,
                tone=PromptTone.EDUCATIONAL,
                variables=["language", "code"],
                tags=["beginner", "educational", "simple"],
            ),
            PromptTemplate(
                id="explain_security",
                name="Security Analysis",
                template="""Analyze this code for security vulnerabilities:

```{language}
{code}
```

Check for:
1. Input validation issues
2. Injection vulnerabilities
3. Authentication/authorization flaws
4. Data exposure risks
5. Security best practices violations""",
                category=PromptCategory.SECURITY,
                variables=["language", "code"],
                tags=["security", "vulnerability", "audit"],
            ),
            PromptTemplate(
                id="explain_performance",
                name="Performance Analysis",
                template="""Analyze the performance of this code:

```{language}
{code}
```

Evaluate:
1. Time complexity
2. Space complexity
3. Potential bottlenecks
4. Optimization opportunities
5. Caching considerations""",
                category=PromptCategory.PERFORMANCE,
                variables=["language", "code"],
                tags=["performance", "optimization", "complexity"],
            ),
            PromptTemplate(
                id="explain_testing",
                name="Testing Guide",
                template="""Suggest tests for this code:

```{language}
{code}
```

Provide:
1. Unit test cases
2. Edge cases to test
3. Mock requirements
4. Integration test considerations
5. Example test code""",
                category=PromptCategory.TESTING,
                variables=["language", "code"],
                tags=["testing", "unit-test", "quality"],
            ),
        ]

        for prompt in builtins:
            self.add_prompt(prompt)


# =============================================================================
# Global Instance Management
# =============================================================================


_global_library: Optional[PromptLibrary] = None


def get_prompt_library() -> PromptLibrary:
    """Get the global prompt library instance."""
    global _global_library
    if _global_library is None:
        _global_library = PromptLibrary()
    return _global_library


def reset_prompt_library() -> None:
    """Reset the global prompt library."""
    global _global_library
    _global_library = None


def create_prompt_library(config: Optional[LibraryConfig] = None) -> PromptLibrary:
    """Create a new prompt library with optional config."""
    return PromptLibrary(config=config)


def get_prompt(prompt_id: str) -> Optional[PromptTemplate]:
    """Convenience function to get a prompt."""
    return get_prompt_library().get_prompt(prompt_id)


def search_prompts(
    query: Optional[str] = None,
    category: Optional[PromptCategory] = None,
    tags: Optional[List[str]] = None,
) -> List[PromptTemplate]:
    """Convenience function to search prompts."""
    return get_prompt_library().search(query=query, category=category, tags=tags)


def render_prompt(prompt_id: str, **kwargs: Any) -> Optional[str]:
    """Convenience function to render a prompt."""
    prompt = get_prompt(prompt_id)
    if prompt:
        return prompt.render(**kwargs)
    return None
