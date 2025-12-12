"""Advanced caching strategies for code explanations.

This module provides sophisticated caching mechanisms including semantic caching,
similarity-based cache lookup, and intelligent cache invalidation strategies.
"""

import hashlib
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .types import CodeElement, CodeElementType, ExplanationResult, ExplainLevel


# =============================================================================
# Enums and Type Definitions
# =============================================================================


class CacheStrategy(str, Enum):
    """Caching strategies for explanations."""

    EXACT = "exact"  # Exact match only
    FUZZY = "fuzzy"  # Allow minor variations
    SEMANTIC = "semantic"  # Semantic similarity
    STRUCTURAL = "structural"  # Code structure based
    CONTENT_HASH = "content_hash"  # Content-based hashing


class InvalidationPolicy(str, Enum):
    """Cache invalidation policies."""

    TTL = "ttl"  # Time-to-live
    LRU = "lru"  # Least recently used
    LFU = "lfu"  # Least frequently used
    FIFO = "fifo"  # First in, first out
    SIZE = "size"  # Size-based eviction
    MANUAL = "manual"  # Manual invalidation only


class CacheHitType(str, Enum):
    """Type of cache hit."""

    EXACT = "exact"
    SIMILAR = "similar"
    PARTIAL = "partial"
    MISS = "miss"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CachedExplanation:
    """A cached explanation entry."""

    explanation: ExplanationResult
    key: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    access_count: int = 0
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = datetime.utcnow().isoformat()
        self.access_count += 1


@dataclass
class CacheLookupResult:
    """Result of a cache lookup."""

    hit: bool
    hit_type: CacheHitType
    explanation: Optional[ExplanationResult] = None
    similarity_score: float = 0.0
    key_used: Optional[str] = None
    lookup_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStatistics:
    """Statistics for cache performance."""

    total_hits: int = 0
    total_misses: int = 0
    exact_hits: int = 0
    similar_hits: int = 0
    partial_hits: int = 0
    total_entries: int = 0
    total_size_bytes: int = 0
    avg_lookup_time_ms: float = 0.0
    evictions: int = 0
    oldest_entry: Optional[str] = None
    newest_entry: Optional[str] = None

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0


@dataclass
class SemanticCacheConfig:
    """Configuration for semantic caching."""

    strategy: CacheStrategy = CacheStrategy.SEMANTIC
    invalidation_policy: InvalidationPolicy = InvalidationPolicy.LRU
    max_entries: int = 1000
    max_size_mb: float = 100.0
    ttl_seconds: int = 3600  # 1 hour default
    similarity_threshold: float = 0.85
    enable_fuzzy_matching: bool = True
    enable_structural_matching: bool = True
    normalize_code: bool = True
    ignore_comments: bool = True
    ignore_whitespace: bool = True
    case_sensitive: bool = False


# =============================================================================
# Key Generation
# =============================================================================


class CacheKeyGenerator:
    """Generates cache keys for code elements."""

    def __init__(self, config: Optional[SemanticCacheConfig] = None):
        """Initialize key generator."""
        self.config = config or SemanticCacheConfig()

    def generate_key(
        self,
        element: CodeElement,
        level: ExplainLevel,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a cache key for an element.

        Args:
            element: Code element to generate key for.
            level: Explanation level.
            context: Optional context for key generation.

        Returns:
            Cache key string.
        """
        code = self._normalize_code(element.source_code)

        components = [
            element.type.value,
            element.name,
            level.value,
            code,
        ]

        if context:
            # Include relevant context in key
            for key in sorted(context.keys()):
                if key in ["model", "temperature", "max_tokens"]:
                    components.append(f"{key}:{context[key]}")

        key_string = "|".join(str(c) for c in components)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]

    def generate_fuzzy_key(self, element: CodeElement, level: ExplainLevel) -> str:
        """Generate a fuzzy cache key that ignores minor variations."""
        code = self._normalize_code(element.source_code)
        # Extract structural signature
        signature = self._extract_signature(code, element.type)

        components = [element.type.value, signature, level.value]
        key_string = "|".join(str(c) for c in components)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]

    def generate_structural_key(self, element: CodeElement) -> str:
        """Generate a structural key based on code structure."""
        structure = self._extract_structure(element.source_code, element.type)
        return hashlib.sha256(structure.encode()).hexdigest()[:32]

    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison."""
        normalized = code

        if self.config.ignore_comments:
            # Remove single-line comments
            normalized = re.sub(r"#.*$", "", normalized, flags=re.MULTILINE)
            normalized = re.sub(r"//.*$", "", normalized, flags=re.MULTILINE)
            # Remove multi-line comments
            normalized = re.sub(r"/\*.*?\*/", "", normalized, flags=re.DOTALL)
            normalized = re.sub(r'""".*?"""', "", normalized, flags=re.DOTALL)
            normalized = re.sub(r"'''.*?'''", "", normalized, flags=re.DOTALL)

        if self.config.ignore_whitespace:
            # Normalize whitespace
            normalized = re.sub(r"\s+", " ", normalized)
            normalized = normalized.strip()

        if not self.config.case_sensitive:
            normalized = normalized.lower()

        return normalized

    def _extract_signature(self, code: str, element_type: CodeElementType) -> str:
        """Extract function/class signature for fuzzy matching."""
        if element_type == CodeElementType.FUNCTION:
            # Extract function signature
            match = re.search(r"def\s+(\w+)\s*\((.*?)\)", code, re.DOTALL)
            if match:
                name = match.group(1)
                params = self._normalize_params(match.group(2))
                return f"func:{name}({params})"

        elif element_type == CodeElementType.CLASS:
            # Extract class signature
            match = re.search(r"class\s+(\w+)(?:\s*\((.*?)\))?", code)
            if match:
                name = match.group(1)
                bases = match.group(2) or ""
                return f"class:{name}({bases})"

        # Fallback: use normalized code hash
        return hashlib.md5(code.encode()).hexdigest()[:16]

    def _normalize_params(self, params: str) -> str:
        """Normalize function parameters."""
        # Remove type annotations and default values
        params = re.sub(r":\s*[^,=]+", "", params)
        params = re.sub(r"=\s*[^,]+", "", params)
        # Extract parameter names
        names = [p.strip().lstrip("*") for p in params.split(",") if p.strip()]
        return ",".join(names)

    def _extract_structure(self, code: str, element_type: CodeElementType) -> str:
        """Extract structural representation of code."""
        lines = code.split("\n")
        structure_parts = []

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Capture structural keywords
            if re.match(r"^\s*(def|class|if|elif|else|for|while|try|except|with)", stripped):
                indent = len(line) - len(line.lstrip())
                keyword = stripped.split()[0].rstrip(":")
                structure_parts.append(f"{indent}:{keyword}")

        return "|".join(structure_parts)


# =============================================================================
# Similarity Calculators
# =============================================================================


class SimilarityCalculator(ABC):
    """Abstract base class for similarity calculation."""

    @abstractmethod
    def calculate(self, code1: str, code2: str) -> float:
        """Calculate similarity between two code strings.

        Args:
            code1: First code string.
            code2: Second code string.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        pass


class TokenSimilarityCalculator(SimilarityCalculator):
    """Calculate similarity based on token overlap."""

    def calculate(self, code1: str, code2: str) -> float:
        """Calculate token-based similarity."""
        tokens1 = set(self._tokenize(code1))
        tokens2 = set(self._tokenize(code2))

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def _tokenize(self, code: str) -> List[str]:
        """Tokenize code into words/identifiers."""
        # Split on non-alphanumeric characters
        tokens = re.findall(r"\b\w+\b", code.lower())
        return tokens


class StructuralSimilarityCalculator(SimilarityCalculator):
    """Calculate similarity based on code structure."""

    def calculate(self, code1: str, code2: str) -> float:
        """Calculate structural similarity."""
        struct1 = self._extract_structure(code1)
        struct2 = self._extract_structure(code2)

        if not struct1 or not struct2:
            return 0.0

        # Compare structural elements
        common = len(set(struct1) & set(struct2))
        total = len(set(struct1) | set(struct2))

        return common / total if total > 0 else 0.0

    def _extract_structure(self, code: str) -> List[str]:
        """Extract structural elements from code."""
        elements = []
        lines = code.split("\n")

        for line in lines:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())

            # Identify structural elements
            if stripped.startswith("def "):
                elements.append(f"def:{indent}")
            elif stripped.startswith("class "):
                elements.append(f"class:{indent}")
            elif stripped.startswith("if "):
                elements.append(f"if:{indent}")
            elif stripped.startswith("for "):
                elements.append(f"for:{indent}")
            elif stripped.startswith("while "):
                elements.append(f"while:{indent}")
            elif stripped.startswith("return"):
                elements.append(f"return:{indent}")

        return elements


class EditDistanceSimilarityCalculator(SimilarityCalculator):
    """Calculate similarity based on edit distance."""

    def calculate(self, code1: str, code2: str) -> float:
        """Calculate edit distance based similarity."""
        # Normalize codes
        code1 = re.sub(r"\s+", " ", code1.strip())
        code2 = re.sub(r"\s+", " ", code2.strip())

        max_len = max(len(code1), len(code2))
        if max_len == 0:
            return 1.0

        distance = self._levenshtein_distance(code1, code2)
        return 1.0 - (distance / max_len)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


class CombinedSimilarityCalculator(SimilarityCalculator):
    """Combines multiple similarity calculators."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize with optional weights."""
        self.calculators = {
            "token": TokenSimilarityCalculator(),
            "structural": StructuralSimilarityCalculator(),
            "edit": EditDistanceSimilarityCalculator(),
        }
        self.weights = weights or {"token": 0.4, "structural": 0.4, "edit": 0.2}

    def calculate(self, code1: str, code2: str) -> float:
        """Calculate weighted combined similarity."""
        total_score = 0.0
        total_weight = 0.0

        for name, calculator in self.calculators.items():
            weight = self.weights.get(name, 0.0)
            if weight > 0:
                score = calculator.calculate(code1, code2)
                total_score += score * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0


# =============================================================================
# Cache Storage
# =============================================================================


class CacheStorage(ABC):
    """Abstract base class for cache storage."""

    @abstractmethod
    def get(self, key: str) -> Optional[CachedExplanation]:
        """Get an entry from cache."""
        pass

    @abstractmethod
    def set(self, key: str, entry: CachedExplanation) -> None:
        """Set an entry in cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete an entry from cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries."""
        pass

    @abstractmethod
    def keys(self) -> List[str]:
        """Get all keys in cache."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get number of entries in cache."""
        pass


class InMemoryStorage(CacheStorage):
    """In-memory cache storage."""

    def __init__(self):
        """Initialize in-memory storage."""
        self._cache: Dict[str, CachedExplanation] = {}

    def get(self, key: str) -> Optional[CachedExplanation]:
        return self._cache.get(key)

    def set(self, key: str, entry: CachedExplanation) -> None:
        self._cache[key] = entry

    def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()

    def keys(self) -> List[str]:
        return list(self._cache.keys())

    def size(self) -> int:
        return len(self._cache)

    def get_all(self) -> Dict[str, CachedExplanation]:
        """Get all entries."""
        return self._cache.copy()


# =============================================================================
# Main Semantic Cache
# =============================================================================


class SemanticCache:
    """Advanced semantic caching for code explanations."""

    def __init__(
        self,
        config: Optional[SemanticCacheConfig] = None,
        storage: Optional[CacheStorage] = None,
    ):
        """Initialize semantic cache.

        Args:
            config: Cache configuration.
            storage: Optional custom storage backend.
        """
        self.config = config or SemanticCacheConfig()
        self.storage = storage or InMemoryStorage()
        self.key_generator = CacheKeyGenerator(self.config)
        self.similarity_calculator = CombinedSimilarityCalculator()
        self.stats = CacheStatistics()
        self._lookup_times: List[float] = []

    def lookup(
        self,
        element: CodeElement,
        level: ExplainLevel,
        context: Optional[Dict[str, Any]] = None,
    ) -> CacheLookupResult:
        """Look up an explanation in the cache.

        Args:
            element: Code element to look up.
            level: Explanation level.
            context: Optional context for lookup.

        Returns:
            CacheLookupResult with hit/miss information.
        """
        start_time = time.time()

        # Try exact match first
        exact_key = self.key_generator.generate_key(element, level, context)
        cached = self.storage.get(exact_key)

        if cached:
            cached.touch()
            lookup_time = (time.time() - start_time) * 1000
            self._record_lookup(lookup_time)
            self.stats.total_hits += 1
            self.stats.exact_hits += 1

            return CacheLookupResult(
                hit=True,
                hit_type=CacheHitType.EXACT,
                explanation=cached.explanation,
                similarity_score=1.0,
                key_used=exact_key,
                lookup_time_ms=lookup_time,
            )

        # Try fuzzy matching if enabled
        if self.config.enable_fuzzy_matching:
            fuzzy_key = self.key_generator.generate_fuzzy_key(element, level)
            cached = self.storage.get(fuzzy_key)

            if cached:
                cached.touch()
                lookup_time = (time.time() - start_time) * 1000
                self._record_lookup(lookup_time)
                self.stats.total_hits += 1
                self.stats.similar_hits += 1

                return CacheLookupResult(
                    hit=True,
                    hit_type=CacheHitType.SIMILAR,
                    explanation=cached.explanation,
                    similarity_score=0.95,
                    key_used=fuzzy_key,
                    lookup_time_ms=lookup_time,
                )

        # Try semantic similarity search
        if self.config.strategy == CacheStrategy.SEMANTIC:
            result = self._semantic_search(element, level)
            if result.hit:
                lookup_time = (time.time() - start_time) * 1000
                self._record_lookup(lookup_time)
                result.lookup_time_ms = lookup_time
                self.stats.total_hits += 1
                self.stats.similar_hits += 1
                return result

        # Cache miss
        lookup_time = (time.time() - start_time) * 1000
        self._record_lookup(lookup_time)
        self.stats.total_misses += 1

        return CacheLookupResult(
            hit=False,
            hit_type=CacheHitType.MISS,
            lookup_time_ms=lookup_time,
        )

    def store(
        self,
        element: CodeElement,
        level: ExplainLevel,
        explanation: ExplanationResult,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store an explanation in the cache.

        Args:
            element: Code element.
            level: Explanation level.
            explanation: Explanation to store.
            context: Optional context for key generation.

        Returns:
            The cache key used.
        """
        # Check if we need to evict
        self._maybe_evict()

        key = self.key_generator.generate_key(element, level, context)

        entry = CachedExplanation(
            explanation=explanation,
            key=key,
            size_bytes=len(explanation.explanation.encode()),
            metadata={
                "element_type": element.type.value,
                "element_name": element.name,
                "level": level.value,
            },
        )

        self.storage.set(key, entry)
        self.stats.total_entries = self.storage.size()
        self.stats.newest_entry = entry.created_at

        # Also store fuzzy key for fuzzy matching
        if self.config.enable_fuzzy_matching:
            fuzzy_key = self.key_generator.generate_fuzzy_key(element, level)
            if fuzzy_key != key:
                self.storage.set(fuzzy_key, entry)

        return key

    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry.

        Args:
            key: Cache key to invalidate.

        Returns:
            True if entry was found and invalidated.
        """
        result = self.storage.delete(key)
        if result:
            self.stats.total_entries = self.storage.size()
            self.stats.evictions += 1
        return result

    def invalidate_by_element(
        self, element: CodeElement, level: Optional[ExplainLevel] = None
    ) -> int:
        """Invalidate all cache entries for an element.

        Args:
            element: Element to invalidate.
            level: Optional specific level to invalidate.

        Returns:
            Number of entries invalidated.
        """
        count = 0
        keys_to_delete = []

        for key in self.storage.keys():
            entry = self.storage.get(key)
            if entry:
                meta = entry.metadata
                if meta.get("element_name") == element.name:
                    if level is None or meta.get("level") == level.value:
                        keys_to_delete.append(key)

        for key in keys_to_delete:
            self.storage.delete(key)
            count += 1

        self.stats.total_entries = self.storage.size()
        self.stats.evictions += count
        return count

    def clear(self) -> None:
        """Clear all cache entries."""
        count = self.storage.size()
        self.storage.clear()
        self.stats.total_entries = 0
        self.stats.evictions += count

    def get_statistics(self) -> CacheStatistics:
        """Get cache statistics."""
        return self.stats

    def _semantic_search(self, element: CodeElement, level: ExplainLevel) -> CacheLookupResult:
        """Search for semantically similar cached explanations."""
        best_match: Optional[Tuple[CachedExplanation, float]] = None

        for key in self.storage.keys():
            entry = self.storage.get(key)
            if not entry:
                continue

            # Check if same type and level
            if (
                entry.metadata.get("element_type") != element.type.value
                or entry.metadata.get("level") != level.value
            ):
                continue

            # Calculate similarity
            cached_code = entry.explanation.element.source_code
            similarity = self.similarity_calculator.calculate(element.source_code, cached_code)

            if similarity >= self.config.similarity_threshold:
                if best_match is None or similarity > best_match[1]:
                    best_match = (entry, similarity)

        if best_match:
            entry, similarity = best_match
            entry.touch()

            return CacheLookupResult(
                hit=True,
                hit_type=CacheHitType.SIMILAR,
                explanation=entry.explanation,
                similarity_score=similarity,
                key_used=entry.key,
            )

        return CacheLookupResult(hit=False, hit_type=CacheHitType.MISS)

    def _maybe_evict(self) -> None:
        """Evict entries if necessary based on policy."""
        if self.storage.size() < self.config.max_entries:
            return

        entries_to_evict = self.storage.size() - self.config.max_entries + 1

        if self.config.invalidation_policy == InvalidationPolicy.LRU:
            self._evict_lru(entries_to_evict)
        elif self.config.invalidation_policy == InvalidationPolicy.LFU:
            self._evict_lfu(entries_to_evict)
        elif self.config.invalidation_policy == InvalidationPolicy.FIFO:
            self._evict_fifo(entries_to_evict)
        elif self.config.invalidation_policy == InvalidationPolicy.TTL:
            self._evict_expired()

    def _evict_lru(self, count: int) -> None:
        """Evict least recently used entries."""
        if isinstance(self.storage, InMemoryStorage):
            entries = list(self.storage.get_all().items())
            entries.sort(key=lambda x: x[1].last_accessed)
            for key, _ in entries[:count]:
                self.storage.delete(key)
                self.stats.evictions += 1

    def _evict_lfu(self, count: int) -> None:
        """Evict least frequently used entries."""
        if isinstance(self.storage, InMemoryStorage):
            entries = list(self.storage.get_all().items())
            entries.sort(key=lambda x: x[1].access_count)
            for key, _ in entries[:count]:
                self.storage.delete(key)
                self.stats.evictions += 1

    def _evict_fifo(self, count: int) -> None:
        """Evict oldest entries first."""
        if isinstance(self.storage, InMemoryStorage):
            entries = list(self.storage.get_all().items())
            entries.sort(key=lambda x: x[1].created_at)
            for key, _ in entries[:count]:
                self.storage.delete(key)
                self.stats.evictions += 1

    def _evict_expired(self) -> None:
        """Evict expired entries based on TTL."""
        now = datetime.utcnow()
        ttl = timedelta(seconds=self.config.ttl_seconds)

        keys_to_delete = []
        for key in self.storage.keys():
            entry = self.storage.get(key)
            if entry:
                created = datetime.fromisoformat(entry.created_at)
                if now - created > ttl:
                    keys_to_delete.append(key)

        for key in keys_to_delete:
            self.storage.delete(key)
            self.stats.evictions += 1

    def _record_lookup(self, time_ms: float) -> None:
        """Record lookup time for statistics."""
        self._lookup_times.append(time_ms)
        # Keep last 1000 lookup times
        if len(self._lookup_times) > 1000:
            self._lookup_times = self._lookup_times[-1000:]
        self.stats.avg_lookup_time_ms = sum(self._lookup_times) / len(self._lookup_times)


# =============================================================================
# Cache Warming
# =============================================================================


class CacheWarmer:
    """Pre-populates cache with common explanations."""

    def __init__(self, cache: SemanticCache):
        """Initialize cache warmer."""
        self.cache = cache

    def warm_from_history(
        self, history: List[Tuple[CodeElement, ExplainLevel, ExplanationResult]]
    ) -> int:
        """Warm cache from historical explanations.

        Args:
            history: List of (element, level, explanation) tuples.

        Returns:
            Number of entries added.
        """
        count = 0
        for element, level, explanation in history:
            self.cache.store(element, level, explanation)
            count += 1
        return count

    def warm_from_file(self, file_path: str) -> int:
        """Warm cache from a JSON file of explanations.

        Args:
            file_path: Path to JSON file.

        Returns:
            Number of entries added.
        """
        import json

        count = 0
        try:
            with open(file_path) as f:
                data = json.load(f)

            for entry in data:
                element = CodeElement(
                    type=CodeElementType(entry["element"]["type"]),
                    name=entry["element"]["name"],
                    source_code=entry["element"]["source_code"],
                    line_start=entry["element"].get("line_start", 0),
                    line_end=entry["element"].get("line_end", 0),
                )
                level = ExplainLevel(entry["level"])
                explanation = ExplanationResult(
                    element=element,
                    explanation=entry["explanation"],
                    level=level,
                    confidence=entry.get("confidence", 0.9),
                )
                self.cache.store(element, level, explanation)
                count += 1

        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

        return count


# =============================================================================
# Global Instance Management
# =============================================================================


_global_semantic_cache: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    """Get the global semantic cache instance."""
    global _global_semantic_cache
    if _global_semantic_cache is None:
        _global_semantic_cache = SemanticCache()
    return _global_semantic_cache


def reset_semantic_cache() -> None:
    """Reset the global semantic cache."""
    global _global_semantic_cache
    _global_semantic_cache = None


def cache_lookup(
    element: CodeElement,
    level: ExplainLevel,
    context: Optional[Dict[str, Any]] = None,
) -> CacheLookupResult:
    """Convenience function for cache lookup."""
    return get_semantic_cache().lookup(element, level, context)


def cache_store(
    element: CodeElement,
    level: ExplainLevel,
    explanation: ExplanationResult,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Convenience function for cache storage."""
    return get_semantic_cache().store(element, level, explanation, context)


def create_semantic_cache(
    config: Optional[SemanticCacheConfig] = None,
) -> SemanticCache:
    """Create a new semantic cache with optional config."""
    return SemanticCache(config=config)
