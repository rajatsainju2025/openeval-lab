"""Code snippet manager for storing and retrieving reusable code snippets.

This module provides functionality to manage code snippets with their
explanations, tags, and metadata for reuse across projects.

Example:
    >>> from openeval.explainers import SnippetManager, create_snippet
    >>> manager = SnippetManager()
    >>> snippet = manager.add_snippet(code="def hello(): pass", title="Hello")
    >>> found = manager.search(tags=["utility"])
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


class SnippetLanguage(Enum):
    """Supported programming languages for snippets."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    RUBY = "ruby"
    PHP = "php"
    CSHARP = "csharp"
    SQL = "sql"
    SHELL = "shell"
    MARKDOWN = "markdown"
    OTHER = "other"


class SnippetVisibility(Enum):
    """Visibility levels for snippets."""

    PRIVATE = "private"
    TEAM = "team"
    PUBLIC = "public"


class SnippetCategory(Enum):
    """Predefined snippet categories."""

    ALGORITHM = "algorithm"
    DATA_STRUCTURE = "data_structure"
    UTILITY = "utility"
    PATTERN = "pattern"
    BOILERPLATE = "boilerplate"
    TEST = "test"
    CONFIGURATION = "configuration"
    DOCUMENTATION = "documentation"
    OTHER = "other"


@dataclass
class SnippetVersion:
    """A version of a code snippet."""

    version_number: int
    code: str
    explanation: str | None
    created_at: datetime
    created_by: str
    changes_description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeSnippet:
    """A reusable code snippet with metadata."""

    snippet_id: str
    title: str
    code: str
    language: SnippetLanguage
    description: str = ""
    explanation: str | None = None
    tags: list[str] = field(default_factory=list)
    category: SnippetCategory = SnippetCategory.OTHER
    visibility: SnippetVisibility = SnippetVisibility.PRIVATE
    author: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    versions: list[SnippetVersion] = field(default_factory=list)
    usage_count: int = 0
    rating: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize snippet after creation."""
        if not self.snippet_id:
            self.snippet_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a unique snippet ID."""
        content = f"{self.title}:{self.code}:{datetime.now().isoformat()}"
        return f"snp_{hashlib.md5(content.encode()).hexdigest()[:12]}"

    def add_version(
        self,
        code: str,
        explanation: str | None = None,
        created_by: str = "",
        changes: str = "",
    ) -> SnippetVersion:
        """Add a new version of the snippet."""
        version = SnippetVersion(
            version_number=self.version,
            code=self.code,
            explanation=self.explanation,
            created_at=self.updated_at,
            created_by=self.author,
            changes_description=changes,
        )
        self.versions.append(version)

        self.code = code
        self.explanation = explanation
        self.version += 1
        self.updated_at = datetime.now()
        if created_by:
            self.author = created_by

        return version

    def get_version(self, version_number: int) -> SnippetVersion | None:
        """Get a specific version of the snippet."""
        for version in self.versions:
            if version.version_number == version_number:
                return version
        return None


@dataclass
class SnippetCollection:
    """A collection of related snippets."""

    collection_id: str
    name: str
    description: str = ""
    snippet_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    author: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchCriteria:
    """Criteria for searching snippets."""

    query: str | None = None
    tags: list[str] | None = None
    languages: list[SnippetLanguage] | None = None
    categories: list[SnippetCategory] | None = None
    visibility: SnippetVisibility | None = None
    author: str | None = None
    min_rating: float | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    limit: int = 50
    offset: int = 0


@dataclass
class SearchResult:
    """Result of a snippet search."""

    snippets: list[CodeSnippet]
    total_count: int
    query_time: float
    criteria: SearchCriteria


class SnippetStorage(ABC):
    """Abstract base class for snippet storage."""

    @abstractmethod
    def save(self, snippet: CodeSnippet) -> None:
        """Save a snippet."""
        pass

    @abstractmethod
    def load(self, snippet_id: str) -> CodeSnippet | None:
        """Load a snippet by ID."""
        pass

    @abstractmethod
    def delete(self, snippet_id: str) -> bool:
        """Delete a snippet."""
        pass

    @abstractmethod
    def list_all(self) -> list[CodeSnippet]:
        """List all snippets."""
        pass

    @abstractmethod
    def search(self, criteria: SearchCriteria) -> list[CodeSnippet]:
        """Search snippets by criteria."""
        pass


class InMemorySnippetStorage(SnippetStorage):
    """In-memory snippet storage implementation."""

    def __init__(self) -> None:
        """Initialize the storage."""
        self._snippets: dict[str, CodeSnippet] = {}

    def save(self, snippet: CodeSnippet) -> None:
        """Save a snippet."""
        self._snippets[snippet.snippet_id] = snippet

    def load(self, snippet_id: str) -> CodeSnippet | None:
        """Load a snippet by ID."""
        return self._snippets.get(snippet_id)

    def delete(self, snippet_id: str) -> bool:
        """Delete a snippet."""
        if snippet_id in self._snippets:
            del self._snippets[snippet_id]
            return True
        return False

    def list_all(self) -> list[CodeSnippet]:
        """List all snippets."""
        return list(self._snippets.values())

    def search(self, criteria: SearchCriteria) -> list[CodeSnippet]:
        """Search snippets by criteria."""
        results = list(self._snippets.values())

        if criteria.query:
            query_lower = criteria.query.lower()
            results = [
                s
                for s in results
                if query_lower in s.title.lower()
                or query_lower in s.code.lower()
                or query_lower in s.description.lower()
            ]

        if criteria.tags:
            results = [s for s in results if any(tag in s.tags for tag in criteria.tags)]

        if criteria.languages:
            results = [s for s in results if s.language in criteria.languages]

        if criteria.categories:
            results = [s for s in results if s.category in criteria.categories]

        if criteria.visibility:
            results = [s for s in results if s.visibility == criteria.visibility]

        if criteria.author:
            results = [s for s in results if s.author == criteria.author]

        if criteria.min_rating is not None:
            results = [s for s in results if s.rating >= criteria.min_rating]

        if criteria.date_from:
            results = [s for s in results if s.created_at >= criteria.date_from]

        if criteria.date_to:
            results = [s for s in results if s.created_at <= criteria.date_to]

        # Apply pagination
        return results[criteria.offset : criteria.offset + criteria.limit]


class FileSnippetStorage(SnippetStorage):
    """File-based snippet storage implementation."""

    def __init__(self, base_path: Path | str) -> None:
        """Initialize with base path."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._index_path = self.base_path / "index.json"
        self._index: dict[str, str] = self._load_index()

    def _load_index(self) -> dict[str, str]:
        """Load the snippet index."""
        if self._index_path.exists():
            with open(self._index_path) as file_handle:
                return json.load(file_handle)
        return {}

    def _save_index(self) -> None:
        """Save the snippet index."""
        with open(self._index_path, "w") as file_handle:
            json.dump(self._index, file_handle)

    def _snippet_path(self, snippet_id: str) -> Path:
        """Get the path for a snippet file."""
        return self.base_path / f"{snippet_id}.json"

    def save(self, snippet: CodeSnippet) -> None:
        """Save a snippet."""
        data = {
            "snippet_id": snippet.snippet_id,
            "title": snippet.title,
            "code": snippet.code,
            "language": snippet.language.value,
            "description": snippet.description,
            "explanation": snippet.explanation,
            "tags": snippet.tags,
            "category": snippet.category.value,
            "visibility": snippet.visibility.value,
            "author": snippet.author,
            "created_at": snippet.created_at.isoformat(),
            "updated_at": snippet.updated_at.isoformat(),
            "version": snippet.version,
            "usage_count": snippet.usage_count,
            "rating": snippet.rating,
            "metadata": snippet.metadata,
        }

        with open(self._snippet_path(snippet.snippet_id), "w") as file_handle:
            json.dump(data, file_handle, indent=2)

        self._index[snippet.snippet_id] = snippet.title
        self._save_index()

    def load(self, snippet_id: str) -> CodeSnippet | None:
        """Load a snippet by ID."""
        path = self._snippet_path(snippet_id)
        if not path.exists():
            return None

        with open(path) as file_handle:
            data = json.load(file_handle)

        return CodeSnippet(
            snippet_id=data["snippet_id"],
            title=data["title"],
            code=data["code"],
            language=SnippetLanguage(data["language"]),
            description=data.get("description", ""),
            explanation=data.get("explanation"),
            tags=data.get("tags", []),
            category=SnippetCategory(data.get("category", "other")),
            visibility=SnippetVisibility(data.get("visibility", "private")),
            author=data.get("author", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            version=data.get("version", 1),
            usage_count=data.get("usage_count", 0),
            rating=data.get("rating", 0.0),
            metadata=data.get("metadata", {}),
        )

    def delete(self, snippet_id: str) -> bool:
        """Delete a snippet."""
        path = self._snippet_path(snippet_id)
        if path.exists():
            path.unlink()
            if snippet_id in self._index:
                del self._index[snippet_id]
                self._save_index()
            return True
        return False

    def list_all(self) -> list[CodeSnippet]:
        """List all snippets."""
        snippets = []
        for snippet_id in self._index:
            snippet = self.load(snippet_id)
            if snippet:
                snippets.append(snippet)
        return snippets

    def search(self, criteria: SearchCriteria) -> list[CodeSnippet]:
        """Search snippets by criteria."""
        all_snippets = self.list_all()

        # Use InMemoryStorage search logic
        memory_storage = InMemorySnippetStorage()
        for snippet in all_snippets:
            memory_storage.save(snippet)

        return memory_storage.search(criteria)


class SnippetManager:
    """Main class for managing code snippets."""

    def __init__(self, storage: SnippetStorage | None = None) -> None:
        """Initialize the snippet manager.

        Args:
            storage: Storage backend for snippets.
        """
        self.storage = storage or InMemorySnippetStorage()
        self._collections: dict[str, SnippetCollection] = {}

    def add_snippet(
        self,
        code: str,
        title: str,
        language: SnippetLanguage = SnippetLanguage.PYTHON,
        description: str = "",
        explanation: str | None = None,
        tags: list[str] | None = None,
        category: SnippetCategory = SnippetCategory.OTHER,
        author: str = "",
        **metadata: Any,
    ) -> CodeSnippet:
        """Add a new code snippet.

        Args:
            code: The code content.
            title: Snippet title.
            language: Programming language.
            description: Brief description.
            explanation: Detailed explanation.
            tags: List of tags.
            category: Snippet category.
            author: Author name.
            **metadata: Additional metadata.

        Returns:
            The created CodeSnippet.
        """
        snippet = CodeSnippet(
            snippet_id="",
            title=title,
            code=code,
            language=language,
            description=description,
            explanation=explanation,
            tags=tags or [],
            category=category,
            author=author,
            metadata=metadata,
        )

        self.storage.save(snippet)
        return snippet

    def get_snippet(self, snippet_id: str) -> CodeSnippet | None:
        """Get a snippet by ID.

        Args:
            snippet_id: The snippet ID.

        Returns:
            The CodeSnippet if found.
        """
        snippet = self.storage.load(snippet_id)
        if snippet:
            snippet.usage_count += 1
            self.storage.save(snippet)
        return snippet

    def update_snippet(
        self,
        snippet_id: str,
        code: str | None = None,
        explanation: str | None = None,
        changes: str = "",
        **updates: Any,
    ) -> CodeSnippet | None:
        """Update a snippet.

        Args:
            snippet_id: The snippet ID.
            code: New code content.
            explanation: New explanation.
            changes: Description of changes.
            **updates: Additional fields to update.

        Returns:
            The updated CodeSnippet.
        """
        snippet = self.storage.load(snippet_id)
        if not snippet:
            return None

        if code is not None:
            snippet.add_version(code, explanation, changes=changes)

        for key, value in updates.items():
            if hasattr(snippet, key):
                setattr(snippet, key, value)

        snippet.updated_at = datetime.now()
        self.storage.save(snippet)
        return snippet

    def delete_snippet(self, snippet_id: str) -> bool:
        """Delete a snippet.

        Args:
            snippet_id: The snippet ID.

        Returns:
            True if deleted successfully.
        """
        return self.storage.delete(snippet_id)

    def search(
        self,
        query: str | None = None,
        tags: list[str] | None = None,
        languages: list[SnippetLanguage] | None = None,
        categories: list[SnippetCategory] | None = None,
        **kwargs: Any,
    ) -> SearchResult:
        """Search for snippets.

        Args:
            query: Text query.
            tags: Filter by tags.
            languages: Filter by languages.
            categories: Filter by categories.
            **kwargs: Additional criteria.

        Returns:
            SearchResult with matching snippets.
        """
        import time

        start = time.time()

        criteria = SearchCriteria(
            query=query,
            tags=tags,
            languages=languages,
            categories=categories,
            **kwargs,
        )

        snippets = self.storage.search(criteria)

        return SearchResult(
            snippets=snippets,
            total_count=len(snippets),
            query_time=time.time() - start,
            criteria=criteria,
        )

    def list_all(self) -> list[CodeSnippet]:
        """List all snippets.

        Returns:
            List of all snippets.
        """
        return self.storage.list_all()

    def iterate(self) -> Iterator[CodeSnippet]:
        """Iterate over all snippets.

        Yields:
            CodeSnippet instances.
        """
        for snippet in self.storage.list_all():
            yield snippet

    def create_collection(
        self,
        name: str,
        description: str = "",
        snippet_ids: list[str] | None = None,
        tags: list[str] | None = None,
        author: str = "",
    ) -> SnippetCollection:
        """Create a snippet collection.

        Args:
            name: Collection name.
            description: Collection description.
            snippet_ids: Initial snippet IDs.
            tags: Collection tags.
            author: Author name.

        Returns:
            The created SnippetCollection.
        """
        collection_id = f"col_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        collection = SnippetCollection(
            collection_id=collection_id,
            name=name,
            description=description,
            snippet_ids=snippet_ids or [],
            tags=tags or [],
            author=author,
        )
        self._collections[collection_id] = collection
        return collection

    def add_to_collection(self, collection_id: str, snippet_id: str) -> bool:
        """Add a snippet to a collection.

        Args:
            collection_id: The collection ID.
            snippet_id: The snippet ID.

        Returns:
            True if added successfully.
        """
        collection = self._collections.get(collection_id)
        if not collection:
            return False

        if snippet_id not in collection.snippet_ids:
            collection.snippet_ids.append(snippet_id)
            collection.updated_at = datetime.now()

        return True

    def get_collection(self, collection_id: str) -> SnippetCollection | None:
        """Get a collection by ID.

        Args:
            collection_id: The collection ID.

        Returns:
            The SnippetCollection if found.
        """
        return self._collections.get(collection_id)

    def get_collection_snippets(self, collection_id: str) -> list[CodeSnippet]:
        """Get all snippets in a collection.

        Args:
            collection_id: The collection ID.

        Returns:
            List of snippets in the collection.
        """
        collection = self._collections.get(collection_id)
        if not collection:
            return []

        snippets = []
        for snippet_id in collection.snippet_ids:
            snippet = self.storage.load(snippet_id)
            if snippet:
                snippets.append(snippet)

        return snippets

    def rate_snippet(self, snippet_id: str, rating: float) -> bool:
        """Rate a snippet.

        Args:
            snippet_id: The snippet ID.
            rating: Rating value (0-5).

        Returns:
            True if rated successfully.
        """
        if not 0 <= rating <= 5:
            return False

        snippet = self.storage.load(snippet_id)
        if not snippet:
            return False

        # Simple average (could be improved with weighted average)
        if snippet.rating == 0:
            snippet.rating = rating
        else:
            snippet.rating = (snippet.rating + rating) / 2

        self.storage.save(snippet)
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get snippet manager statistics.

        Returns:
            Statistics dictionary.
        """
        all_snippets = self.storage.list_all()

        by_language: dict[str, int] = {}
        by_category: dict[str, int] = {}
        total_usage = 0

        for snippet in all_snippets:
            lang = snippet.language.value
            by_language[lang] = by_language.get(lang, 0) + 1

            cat = snippet.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

            total_usage += snippet.usage_count

        return {
            "total_snippets": len(all_snippets),
            "total_collections": len(self._collections),
            "total_usage": total_usage,
            "by_language": by_language,
            "by_category": by_category,
        }

    def detect_language(self, code: str) -> SnippetLanguage:
        """Detect the language of code.

        Args:
            code: Source code.

        Returns:
            Detected SnippetLanguage.
        """
        patterns = {
            SnippetLanguage.PYTHON: [
                r"^\s*def\s+\w+",
                r"^\s*class\s+\w+",
                r"^\s*import\s+",
                r"^\s*from\s+\w+\s+import",
            ],
            SnippetLanguage.JAVASCRIPT: [
                r"^\s*function\s+\w+",
                r"^\s*const\s+\w+\s*=",
                r"^\s*let\s+\w+\s*=",
                r"=>\s*\{",
            ],
            SnippetLanguage.TYPESCRIPT: [
                r":\s*(string|number|boolean|any)\s*[=;]",
                r"interface\s+\w+",
                r"type\s+\w+\s*=",
            ],
            SnippetLanguage.JAVA: [
                r"^\s*public\s+class",
                r"^\s*private\s+\w+",
                r"System\.out\.println",
            ],
            SnippetLanguage.GO: [
                r"^\s*func\s+\w+",
                r"^\s*package\s+\w+",
                r":=\s*",
            ],
            SnippetLanguage.RUST: [
                r"^\s*fn\s+\w+",
                r"^\s*let\s+mut\s+",
                r"impl\s+\w+",
            ],
        }

        for language, lang_patterns in patterns.items():
            for pattern in lang_patterns:
                if re.search(pattern, code, re.MULTILINE):
                    return language

        return SnippetLanguage.OTHER


# Global instance
_snippet_manager: SnippetManager | None = None


def get_snippet_manager() -> SnippetManager:
    """Get the global snippet manager.

    Returns:
        The global SnippetManager instance.
    """
    global _snippet_manager
    if _snippet_manager is None:
        _snippet_manager = SnippetManager()
    return _snippet_manager


def reset_snippet_manager() -> None:
    """Reset the global snippet manager."""
    global _snippet_manager
    _snippet_manager = None


def create_snippet_manager(storage: SnippetStorage | None = None) -> SnippetManager:
    """Create a new snippet manager.

    Args:
        storage: Storage backend.

    Returns:
        New SnippetManager instance.
    """
    return SnippetManager(storage=storage)


def create_snippet(
    code: str,
    title: str,
    language: SnippetLanguage = SnippetLanguage.PYTHON,
    **kwargs: Any,
) -> CodeSnippet:
    """Create a new snippet.

    Args:
        code: The code content.
        title: Snippet title.
        language: Programming language.
        **kwargs: Additional options.

    Returns:
        The created CodeSnippet.
    """
    return get_snippet_manager().add_snippet(code=code, title=title, language=language, **kwargs)


def search_snippets(
    query: str | None = None, tags: list[str] | None = None, **kwargs: Any
) -> SearchResult:
    """Search for snippets.

    Args:
        query: Text query.
        tags: Filter by tags.
        **kwargs: Additional criteria.

    Returns:
        SearchResult with matching snippets.
    """
    return get_snippet_manager().search(query=query, tags=tags, **kwargs)


def create_file_storage(path: str | Path) -> FileSnippetStorage:
    """Create a file-based storage.

    Args:
        path: Base directory path.

    Returns:
        FileSnippetStorage instance.
    """
    return FileSnippetStorage(path)
