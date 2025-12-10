"""Explanation versioning system.

Tracks versions of explanations for code changes and comparison.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .types import CodeElement, ExplainLevel, ExplanationResult


@dataclass
class ExplanationVersion:
    """A versioned snapshot of an explanation."""

    version_id: str
    explanation_result: ExplanationResult
    code_hash: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    parent_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "code_hash": self.code_hash,
            "timestamp": self.timestamp.isoformat(),
            "parent_version": self.parent_version,
            "metadata": self.metadata,
            "tags": self.tags,
            "explanation": self.explanation_result.explanation,
            "confidence": self.explanation_result.confidence,
            "level": self.explanation_result.level.value,
        }


def _compute_code_hash(code: str) -> str:
    """Compute hash of code for change detection.

    Args:
        code: Source code string.

    Returns:
        SHA256 hash of code.
    """
    return hashlib.sha256(code.encode()).hexdigest()[:16]


def _generate_version_id(element_name: str, timestamp: datetime) -> str:
    """Generate unique version ID.

    Args:
        element_name: Name of code element.
        timestamp: Version timestamp.

    Returns:
        Unique version ID string.
    """
    ts_str = timestamp.strftime("%Y%m%d%H%M%S%f")
    hash_input = f"{element_name}:{ts_str}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:12]


class VersionTracker:
    """Tracks explanation versions for code elements.

    Maintains version history and enables comparison across versions.
    """

    def __init__(self, max_versions_per_element: int = 100) -> None:
        """Initialize version tracker.

        Args:
            max_versions_per_element: Maximum versions to keep per element.
        """
        self.max_versions = max_versions_per_element
        # element_name -> [versions] (newest first)
        self._versions: Dict[str, List[ExplanationVersion]] = {}
        # version_id -> ExplanationVersion
        self._version_index: Dict[str, ExplanationVersion] = {}

    def create_version(
        self,
        explanation_result: ExplanationResult,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> ExplanationVersion:
        """Create a new version from an explanation result.

        Args:
            explanation_result: The explanation result to version.
            metadata: Optional metadata for this version.
            tags: Optional tags for this version.

        Returns:
            Created ExplanationVersion.
        """
        element = explanation_result.element
        now = datetime.utcnow()

        # Compute code hash
        code_hash = _compute_code_hash(element.source_code)

        # Generate version ID
        version_id = _generate_version_id(element.name, now)

        # Get parent version (most recent if exists)
        parent_version = None
        if element.name in self._versions and self._versions[element.name]:
            parent_version = self._versions[element.name][0].version_id

        # Create version
        version = ExplanationVersion(
            version_id=version_id,
            explanation_result=explanation_result,
            code_hash=code_hash,
            timestamp=now,
            parent_version=parent_version,
            metadata=metadata or {},
            tags=tags or [],
        )

        # Store version
        if element.name not in self._versions:
            self._versions[element.name] = []

        self._versions[element.name].insert(0, version)
        self._version_index[version_id] = version

        # Trim old versions
        if len(self._versions[element.name]) > self.max_versions:
            removed = self._versions[element.name].pop()
            self._version_index.pop(removed.version_id, None)

        return version

    def get_version(self, version_id: str) -> Optional[ExplanationVersion]:
        """Get a specific version by ID.

        Args:
            version_id: Version ID to retrieve.

        Returns:
            ExplanationVersion or None if not found.
        """
        return self._version_index.get(version_id)

    def get_latest(self, element_name: str) -> Optional[ExplanationVersion]:
        """Get the latest version for an element.

        Args:
            element_name: Name of the code element.

        Returns:
            Latest ExplanationVersion or None if no versions.
        """
        versions = self._versions.get(element_name, [])
        return versions[0] if versions else None

    def get_history(
        self,
        element_name: str,
        limit: Optional[int] = None,
    ) -> List[ExplanationVersion]:
        """Get version history for an element.

        Args:
            element_name: Name of the code element.
            limit: Maximum number of versions to return.

        Returns:
            List of versions (newest first).
        """
        versions = self._versions.get(element_name, [])
        if limit:
            versions = versions[:limit]
        return versions

    def get_by_code_hash(
        self,
        element_name: str,
        code_hash: str,
    ) -> Optional[ExplanationVersion]:
        """Find version matching a specific code hash.

        Useful for finding cached explanation for unchanged code.

        Args:
            element_name: Name of the code element.
            code_hash: Hash of the code to find.

        Returns:
            Matching ExplanationVersion or None.
        """
        for version in self._versions.get(element_name, []):
            if version.code_hash == code_hash:
                return version
        return None

    def find_by_code(
        self,
        element_name: str,
        code: str,
    ) -> Optional[ExplanationVersion]:
        """Find version matching specific code.

        Args:
            element_name: Name of the code element.
            code: Source code to match.

        Returns:
            Matching ExplanationVersion or None.
        """
        code_hash = _compute_code_hash(code)
        return self.get_by_code_hash(element_name, code_hash)

    def get_by_tag(self, tag: str) -> List[ExplanationVersion]:
        """Get all versions with a specific tag.

        Args:
            tag: Tag to search for.

        Returns:
            List of matching versions.
        """
        results = []
        for version in self._version_index.values():
            if tag in version.tags:
                results.append(version)
        return results

    def tag_version(self, version_id: str, tag: str) -> bool:
        """Add a tag to a version.

        Args:
            version_id: Version to tag.
            tag: Tag to add.

        Returns:
            True if tagged, False if version not found.
        """
        version = self._version_index.get(version_id)
        if version and tag not in version.tags:
            version.tags.append(tag)
            return True
        return False

    def untag_version(self, version_id: str, tag: str) -> bool:
        """Remove a tag from a version.

        Args:
            version_id: Version to untag.
            tag: Tag to remove.

        Returns:
            True if untagged, False if version not found or tag missing.
        """
        version = self._version_index.get(version_id)
        if version and tag in version.tags:
            version.tags.remove(tag)
            return True
        return False

    def compare_versions(
        self,
        version_id_1: str,
        version_id_2: str,
    ) -> Optional[Dict[str, Any]]:
        """Compare two versions.

        Args:
            version_id_1: First version ID.
            version_id_2: Second version ID.

        Returns:
            Comparison dictionary or None if versions not found.
        """
        v1 = self.get_version(version_id_1)
        v2 = self.get_version(version_id_2)

        if not v1 or not v2:
            return None

        return {
            "version_1": version_id_1,
            "version_2": version_id_2,
            "code_changed": v1.code_hash != v2.code_hash,
            "time_diff_seconds": abs((v1.timestamp - v2.timestamp).total_seconds()),
            "confidence_diff": v1.explanation_result.confidence - v2.explanation_result.confidence,
            "explanation_length_1": len(v1.explanation_result.explanation),
            "explanation_length_2": len(v2.explanation_result.explanation),
            "explanation_length_diff": len(v1.explanation_result.explanation)
            - len(v2.explanation_result.explanation),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get version tracker statistics.

        Returns:
            Dictionary with statistics.
        """
        total_versions = len(self._version_index)
        elements = len(self._versions)

        return {
            "total_versions": total_versions,
            "tracked_elements": elements,
            "avg_versions_per_element": total_versions / elements if elements > 0 else 0,
            "max_versions_limit": self.max_versions,
        }

    def clear(self, element_name: Optional[str] = None) -> int:
        """Clear version history.

        Args:
            element_name: Specific element to clear (all if None).

        Returns:
            Number of versions cleared.
        """
        if element_name:
            versions = self._versions.pop(element_name, [])
            for v in versions:
                self._version_index.pop(v.version_id, None)
            return len(versions)
        else:
            count = len(self._version_index)
            self._versions.clear()
            self._version_index.clear()
            return count


class VersionedExplainer:
    """Wrapper that adds versioning to any explainer."""

    def __init__(
        self,
        explainer: Any,  # CodeExplainer
        tracker: Optional[VersionTracker] = None,
    ) -> None:
        """Initialize versioned explainer.

        Args:
            explainer: CodeExplainer to wrap.
            tracker: VersionTracker to use (creates new if None).
        """
        self.explainer = explainer
        self.tracker = tracker or VersionTracker()

    def explain(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.DETAILED,
        context: Optional[Dict[str, Any]] = None,
        use_cached: bool = True,
        tags: Optional[List[str]] = None,
    ) -> ExplanationResult:
        """Explain with automatic versioning.

        Args:
            element: Code element to explain.
            level: Explanation detail level.
            context: Additional context.
            use_cached: Whether to use cached version if code unchanged.
            tags: Tags to add to created version.

        Returns:
            ExplanationResult.
        """
        # Check for cached version with same code
        if use_cached:
            cached = self.tracker.find_by_code(element.name, element.source_code)
            if cached:
                return cached.explanation_result

        # Generate new explanation
        result = self.explainer.explain(element, level, context)

        # Create version
        self.tracker.create_version(result, tags=tags)

        return result

    def get_history(
        self,
        element_name: str,
        limit: Optional[int] = None,
    ) -> List[ExplanationVersion]:
        """Get explanation history for an element.

        Args:
            element_name: Name of code element.
            limit: Maximum versions to return.

        Returns:
            List of versions.
        """
        return self.tracker.get_history(element_name, limit)

    def get_latest_version(self, element_name: str) -> Optional[ExplanationVersion]:
        """Get latest version for an element.

        Args:
            element_name: Name of code element.

        Returns:
            Latest version or None.
        """
        return self.tracker.get_latest(element_name)


# Global version tracker
_global_version_tracker = VersionTracker()


def get_version_tracker() -> VersionTracker:
    """Get the global version tracker instance.

    Returns:
        VersionTracker singleton.
    """
    return _global_version_tracker
