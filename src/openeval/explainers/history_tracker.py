"""History tracker for explanation versioning and change tracking.

This module provides tools for tracking explanation history, managing
versions, and enabling rollback to previous explanations.
"""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .types import CodeElement, ExplanationResult


# =============================================================================
# Enums and Type Definitions
# =============================================================================


class ChangeType(str, Enum):
    """Types of changes to explanations."""

    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    RESTORED = "restored"
    MERGED = "merged"


class VersionStatus(str, Enum):
    """Status of a version."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    DRAFT = "draft"


class DiffType(str, Enum):
    """Types of differences between versions."""

    ADDITION = "addition"
    DELETION = "deletion"
    MODIFICATION = "modification"
    NO_CHANGE = "no_change"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class HistoryEntry:
    """A single entry in the explanation history."""

    id: str
    version: int
    element_id: str
    explanation_text: str
    timestamp: str
    change_type: ChangeType
    author: str = "system"
    parent_version: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    @property
    def content_hash(self) -> str:
        """Get hash of the explanation content."""
        return hashlib.sha256(self.explanation_text.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "version": self.version,
            "element_id": self.element_id,
            "explanation_text": self.explanation_text,
            "timestamp": self.timestamp,
            "change_type": self.change_type.value,
            "author": self.author,
            "parent_version": self.parent_version,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoryEntry":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            version=data["version"],
            element_id=data["element_id"],
            explanation_text=data["explanation_text"],
            timestamp=data["timestamp"],
            change_type=ChangeType(data["change_type"]),
            author=data.get("author", "system"),
            parent_version=data.get("parent_version"),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
        )


@dataclass
class VersionDiff:
    """Difference between two versions."""

    from_version: int
    to_version: int
    diff_type: DiffType
    added_lines: List[str] = field(default_factory=list)
    removed_lines: List[str] = field(default_factory=list)
    changed_sections: List[Dict[str, Any]] = field(default_factory=list)
    similarity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "diff_type": self.diff_type.value,
            "added_lines": self.added_lines,
            "removed_lines": self.removed_lines,
            "changed_sections": self.changed_sections,
            "similarity_score": self.similarity_score,
        }


@dataclass
class Branch:
    """A branch of explanation history."""

    name: str
    base_version: int
    head_version: int
    created_at: str
    author: str = "system"
    description: str = ""
    is_merged: bool = False
    merged_into: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "base_version": self.base_version,
            "head_version": self.head_version,
            "created_at": self.created_at,
            "author": self.author,
            "description": self.description,
            "is_merged": self.is_merged,
            "merged_into": self.merged_into,
        }


@dataclass
class HistoryConfig:
    """Configuration for history tracker."""

    max_versions: int = 100
    auto_archive_after: int = 50
    enable_branching: bool = True
    track_metadata: bool = True
    compress_old_versions: bool = False
    retention_days: Optional[int] = None


@dataclass
class HistoryStats:
    """Statistics about history."""

    total_versions: int = 0
    total_elements: int = 0
    total_changes: int = 0
    oldest_entry: Optional[str] = None
    newest_entry: Optional[str] = None
    most_changed_element: Optional[str] = None
    change_frequency: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_versions": self.total_versions,
            "total_elements": self.total_elements,
            "total_changes": self.total_changes,
            "oldest_entry": self.oldest_entry,
            "newest_entry": self.newest_entry,
            "most_changed_element": self.most_changed_element,
            "change_frequency": self.change_frequency,
        }


# =============================================================================
# Storage Backends
# =============================================================================


class HistoryStorage(ABC):
    """Abstract base class for history storage."""

    @abstractmethod
    def save(self, entry: HistoryEntry) -> None:
        """Save a history entry."""
        pass

    @abstractmethod
    def get(self, element_id: str, version: Optional[int] = None) -> Optional[HistoryEntry]:
        """Get a history entry."""
        pass

    @abstractmethod
    def list_versions(self, element_id: str) -> List[HistoryEntry]:
        """List all versions for an element."""
        pass

    @abstractmethod
    def delete(self, element_id: str, version: int) -> bool:
        """Delete a specific version."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all history."""
        pass


class InMemoryHistoryStorage(HistoryStorage):
    """In-memory storage for history."""

    def __init__(self):
        self._entries: Dict[str, Dict[int, HistoryEntry]] = {}

    def save(self, entry: HistoryEntry) -> None:
        """Save a history entry."""
        if entry.element_id not in self._entries:
            self._entries[entry.element_id] = {}
        self._entries[entry.element_id][entry.version] = entry

    def get(self, element_id: str, version: Optional[int] = None) -> Optional[HistoryEntry]:
        """Get a history entry."""
        if element_id not in self._entries:
            return None

        versions = self._entries[element_id]
        if version is None:
            # Get latest version
            if not versions:
                return None
            latest = max(versions.keys())
            return versions[latest]
        return versions.get(version)

    def list_versions(self, element_id: str) -> List[HistoryEntry]:
        """List all versions for an element."""
        if element_id not in self._entries:
            return []
        return sorted(self._entries[element_id].values(), key=lambda e: e.version)

    def delete(self, element_id: str, version: int) -> bool:
        """Delete a specific version."""
        if element_id in self._entries and version in self._entries[element_id]:
            del self._entries[element_id][version]
            return True
        return False

    def clear(self) -> None:
        """Clear all history."""
        self._entries.clear()

    def get_all_elements(self) -> List[str]:
        """Get all tracked element IDs."""
        return list(self._entries.keys())


class FileHistoryStorage(HistoryStorage):
    """File-based storage for history."""

    def __init__(self, base_path: str = ".history"):
        self._base_path = base_path
        self._index: Dict[str, Dict[int, str]] = {}

    def save(self, entry: HistoryEntry) -> None:
        """Save a history entry."""
        if entry.element_id not in self._index:
            self._index[entry.element_id] = {}

        filename = f"{entry.element_id}_{entry.version}.json"
        self._index[entry.element_id][entry.version] = filename

        # In real implementation, write to file
        # For now, just track in memory

    def get(self, element_id: str, version: Optional[int] = None) -> Optional[HistoryEntry]:
        """Get a history entry."""
        # Simplified - in real implementation would read from file
        return None

    def list_versions(self, element_id: str) -> List[HistoryEntry]:
        """List all versions for an element."""
        return []

    def delete(self, element_id: str, version: int) -> bool:
        """Delete a specific version."""
        if element_id in self._index and version in self._index[element_id]:
            del self._index[element_id][version]
            return True
        return False

    def clear(self) -> None:
        """Clear all history."""
        self._index.clear()


# =============================================================================
# Diff Engine
# =============================================================================


class DiffEngine:
    """Engine for computing differences between versions."""

    def compute_diff(self, old_text: str, new_text: str) -> VersionDiff:
        """Compute diff between two text versions.

        Args:
            old_text: Original text.
            new_text: New text.

        Returns:
            VersionDiff with details of changes.
        """
        old_lines = old_text.split("\n")
        new_lines = new_text.split("\n")

        added = []
        removed = []

        old_set = set(old_lines)
        new_set = set(new_lines)

        for line in new_lines:
            if line not in old_set:
                added.append(line)

        for line in old_lines:
            if line not in new_set:
                removed.append(line)

        # Determine diff type
        if not added and not removed:
            diff_type = DiffType.NO_CHANGE
        elif not removed:
            diff_type = DiffType.ADDITION
        elif not added:
            diff_type = DiffType.DELETION
        else:
            diff_type = DiffType.MODIFICATION

        # Calculate similarity
        similarity = self._calculate_similarity(old_text, new_text)

        return VersionDiff(
            from_version=0,
            to_version=0,
            diff_type=diff_type,
            added_lines=added,
            removed_lines=removed,
            similarity_score=similarity,
        )

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


# =============================================================================
# Main History Tracker
# =============================================================================


class HistoryTracker:
    """Main history tracker for explanations."""

    def __init__(
        self,
        storage: Optional[HistoryStorage] = None,
        config: Optional[HistoryConfig] = None,
    ):
        """Initialize history tracker.

        Args:
            storage: Storage backend.
            config: Configuration options.
        """
        self._storage = storage or InMemoryHistoryStorage()
        self._config = config or HistoryConfig()
        self._diff_engine = DiffEngine()
        self._branches: Dict[str, Branch] = {}
        self._current_branch = "main"

    def track(
        self,
        element: CodeElement,
        explanation: ExplanationResult,
        author: str = "system",
        tags: Optional[List[str]] = None,
    ) -> HistoryEntry:
        """Track a new explanation version.

        Args:
            element: Code element being explained.
            explanation: Explanation result.
            author: Author of the change.
            tags: Optional tags for the version.

        Returns:
            The created history entry.
        """
        element_id = self._get_element_id(element)
        existing = self._storage.list_versions(element_id)

        version = len(existing) + 1
        parent_version = existing[-1].version if existing else None

        # Determine change type
        if not existing:
            change_type = ChangeType.CREATED
        else:
            change_type = ChangeType.UPDATED

        entry = HistoryEntry(
            id=f"{element_id}_v{version}",
            version=version,
            element_id=element_id,
            explanation_text=explanation.explanation,
            timestamp=datetime.utcnow().isoformat(),
            change_type=change_type,
            author=author,
            parent_version=parent_version,
            metadata={
                "level": explanation.level.value,
                "confidence": explanation.confidence,
                "model_used": explanation.model_used,
                "branch": self._current_branch,
            },
            tags=tags or [],
        )

        self._storage.save(entry)

        # Archive old versions if needed
        self._auto_archive(element_id)

        return entry

    def get_version(
        self, element: CodeElement, version: Optional[int] = None
    ) -> Optional[HistoryEntry]:
        """Get a specific version of an explanation.

        Args:
            element: Code element.
            version: Version number (None for latest).

        Returns:
            History entry if found.
        """
        element_id = self._get_element_id(element)
        return self._storage.get(element_id, version)

    def get_history(self, element: CodeElement) -> List[HistoryEntry]:
        """Get full history for an element.

        Args:
            element: Code element.

        Returns:
            List of history entries.
        """
        element_id = self._get_element_id(element)
        return self._storage.list_versions(element_id)

    def rollback(
        self, element: CodeElement, to_version: int, author: str = "system"
    ) -> Optional[HistoryEntry]:
        """Rollback to a previous version.

        Args:
            element: Code element.
            to_version: Version to rollback to.
            author: Author of the rollback.

        Returns:
            New history entry representing the rollback.
        """
        element_id = self._get_element_id(element)
        target = self._storage.get(element_id, to_version)

        if not target:
            return None

        existing = self._storage.list_versions(element_id)
        new_version = len(existing) + 1

        entry = HistoryEntry(
            id=f"{element_id}_v{new_version}",
            version=new_version,
            element_id=element_id,
            explanation_text=target.explanation_text,
            timestamp=datetime.utcnow().isoformat(),
            change_type=ChangeType.RESTORED,
            author=author,
            parent_version=existing[-1].version if existing else None,
            metadata={
                "restored_from": to_version,
                "branch": self._current_branch,
            },
            tags=["rollback"],
        )

        self._storage.save(entry)
        return entry

    def compare_versions(
        self, element: CodeElement, version1: int, version2: int
    ) -> Optional[VersionDiff]:
        """Compare two versions of an explanation.

        Args:
            element: Code element.
            version1: First version.
            version2: Second version.

        Returns:
            VersionDiff if both versions exist.
        """
        element_id = self._get_element_id(element)
        entry1 = self._storage.get(element_id, version1)
        entry2 = self._storage.get(element_id, version2)

        if not entry1 or not entry2:
            return None

        diff = self._diff_engine.compute_diff(entry1.explanation_text, entry2.explanation_text)
        diff.from_version = version1
        diff.to_version = version2

        return diff

    def create_branch(
        self,
        name: str,
        element: CodeElement,
        author: str = "system",
        description: str = "",
    ) -> Optional[Branch]:
        """Create a new branch for an element.

        Args:
            name: Branch name.
            element: Code element.
            author: Branch author.
            description: Branch description.

        Returns:
            Created branch.
        """
        if not self._config.enable_branching:
            return None

        element_id = self._get_element_id(element)
        current = self._storage.get(element_id)

        if not current:
            return None

        branch_key = f"{element_id}:{name}"
        if branch_key in self._branches:
            return None  # Branch already exists

        branch = Branch(
            name=name,
            base_version=current.version,
            head_version=current.version,
            created_at=datetime.utcnow().isoformat(),
            author=author,
            description=description,
        )

        self._branches[branch_key] = branch
        return branch

    def switch_branch(self, name: str) -> bool:
        """Switch to a different branch.

        Args:
            name: Branch name.

        Returns:
            True if successful.
        """
        self._current_branch = name
        return True

    def merge_branch(
        self,
        element: CodeElement,
        source_branch: str,
        target_branch: str = "main",
        author: str = "system",
    ) -> Optional[HistoryEntry]:
        """Merge one branch into another.

        Args:
            element: Code element.
            source_branch: Branch to merge from.
            target_branch: Branch to merge into.
            author: Author of the merge.

        Returns:
            New history entry for the merge.
        """
        element_id = self._get_element_id(element)
        branch_key = f"{element_id}:{source_branch}"

        if branch_key not in self._branches:
            return None

        branch = self._branches[branch_key]
        source_entry = self._storage.get(element_id, branch.head_version)

        if not source_entry:
            return None

        existing = self._storage.list_versions(element_id)
        new_version = len(existing) + 1

        entry = HistoryEntry(
            id=f"{element_id}_v{new_version}",
            version=new_version,
            element_id=element_id,
            explanation_text=source_entry.explanation_text,
            timestamp=datetime.utcnow().isoformat(),
            change_type=ChangeType.MERGED,
            author=author,
            parent_version=existing[-1].version if existing else None,
            metadata={
                "merged_from": source_branch,
                "merged_into": target_branch,
                "source_version": branch.head_version,
                "branch": target_branch,
            },
            tags=["merge"],
        )

        self._storage.save(entry)

        # Mark branch as merged
        branch.is_merged = True
        branch.merged_into = target_branch

        return entry

    def tag_version(self, element: CodeElement, version: int, tag: str) -> bool:
        """Add a tag to a specific version.

        Args:
            element: Code element.
            version: Version number.
            tag: Tag to add.

        Returns:
            True if successful.
        """
        element_id = self._get_element_id(element)
        entry = self._storage.get(element_id, version)

        if not entry:
            return False

        if tag not in entry.tags:
            entry.tags.append(tag)
            self._storage.save(entry)

        return True

    def find_by_tag(self, element: CodeElement, tag: str) -> List[HistoryEntry]:
        """Find versions with a specific tag.

        Args:
            element: Code element.
            tag: Tag to search for.

        Returns:
            List of matching entries.
        """
        element_id = self._get_element_id(element)
        all_versions = self._storage.list_versions(element_id)
        return [e for e in all_versions if tag in e.tags]

    def get_stats(self) -> HistoryStats:
        """Get statistics about the history.

        Returns:
            History statistics.
        """
        if isinstance(self._storage, InMemoryHistoryStorage):
            elements = self._storage.get_all_elements()
        else:
            elements = []

        total_versions = 0
        change_frequency: Dict[str, int] = {}
        oldest: Optional[str] = None
        newest: Optional[str] = None

        for element_id in elements:
            versions = self._storage.list_versions(element_id)
            total_versions += len(versions)
            change_frequency[element_id] = len(versions)

            for entry in versions:
                if oldest is None or entry.timestamp < oldest:
                    oldest = entry.timestamp
                if newest is None or entry.timestamp > newest:
                    newest = entry.timestamp

        most_changed = None
        if change_frequency:
            most_changed = max(change_frequency, key=change_frequency.get)

        return HistoryStats(
            total_versions=total_versions,
            total_elements=len(elements),
            total_changes=total_versions,
            oldest_entry=oldest,
            newest_entry=newest,
            most_changed_element=most_changed,
            change_frequency=change_frequency,
        )

    def clear_history(self, element: Optional[CodeElement] = None) -> None:
        """Clear history for an element or all elements.

        Args:
            element: Specific element to clear (None for all).
        """
        if element:
            element_id = self._get_element_id(element)
            versions = self._storage.list_versions(element_id)
            for entry in versions:
                self._storage.delete(element_id, entry.version)
        else:
            self._storage.clear()
            self._branches.clear()

    def _get_element_id(self, element: CodeElement) -> str:
        """Generate unique ID for an element."""
        return f"{element.type.value}:{element.name}:{element.line_start}"

    def _auto_archive(self, element_id: str) -> None:
        """Auto-archive old versions if needed."""
        versions = self._storage.list_versions(element_id)
        if len(versions) > self._config.max_versions:
            # Archive or delete oldest versions
            to_remove = len(versions) - self._config.max_versions
            for entry in versions[:to_remove]:
                self._storage.delete(element_id, entry.version)


# =============================================================================
# Global Instance Management
# =============================================================================


_global_tracker: Optional[HistoryTracker] = None


def get_history_tracker() -> HistoryTracker:
    """Get the global history tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = HistoryTracker()
    return _global_tracker


def reset_history_tracker() -> None:
    """Reset the global history tracker."""
    global _global_tracker
    _global_tracker = None


def create_history_tracker(
    storage: Optional[HistoryStorage] = None,
    config: Optional[HistoryConfig] = None,
) -> HistoryTracker:
    """Create a new history tracker with optional config."""
    return HistoryTracker(storage=storage, config=config)


def track_explanation(
    element: CodeElement,
    explanation: ExplanationResult,
    author: str = "system",
) -> HistoryEntry:
    """Convenience function to track an explanation."""
    return get_history_tracker().track(element, explanation, author)


def get_explanation_history(element: CodeElement) -> List[HistoryEntry]:
    """Convenience function to get explanation history."""
    return get_history_tracker().get_history(element)


def rollback_explanation(element: CodeElement, to_version: int) -> Optional[HistoryEntry]:
    """Convenience function to rollback an explanation."""
    return get_history_tracker().rollback(element, to_version)
