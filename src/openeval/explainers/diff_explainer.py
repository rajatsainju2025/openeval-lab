"""Explanation diff system for comparing explanations.

Enables comparison of explanations across code versions.
"""

import difflib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .types import ExplanationResult


class DiffType(str, Enum):
    """Type of difference detected."""

    UNCHANGED = "unchanged"
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


class ChangeSignificance(str, Enum):
    """Significance level of changes."""

    NONE = "none"
    MINOR = "minor"  # Formatting, small wording changes
    MODERATE = "moderate"  # Structure changes, new sections
    MAJOR = "major"  # Completely different explanation


@dataclass
class DiffLine:
    """A single line in a diff."""

    line_number: int
    content: str
    diff_type: DiffType
    original_line: Optional[int] = None


@dataclass
class DiffSection:
    """A section of related diff changes."""

    start_line: int
    end_line: int
    diff_type: DiffType
    original_content: str
    new_content: str


@dataclass
class ExplanationDiff:
    """Detailed diff between two explanations."""

    element_name: str
    old_explanation: str
    new_explanation: str
    code_changed: bool
    diff_lines: List[DiffLine]
    sections: List[DiffSection]
    similarity_ratio: float
    significance: ChangeSignificance
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_changes(self) -> bool:
        """Check if there are any differences."""
        return self.similarity_ratio < 1.0

    @property
    def added_lines(self) -> int:
        """Count of added lines."""
        return sum(1 for line in self.diff_lines if line.diff_type == DiffType.ADDED)

    @property
    def removed_lines(self) -> int:
        """Count of removed lines."""
        return sum(1 for line in self.diff_lines if line.diff_type == DiffType.REMOVED)

    @property
    def modified_sections(self) -> int:
        """Count of modified sections."""
        return len(self.sections)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "element_name": self.element_name,
            "code_changed": self.code_changed,
            "similarity_ratio": self.similarity_ratio,
            "significance": self.significance.value,
            "added_lines": self.added_lines,
            "removed_lines": self.removed_lines,
            "modified_sections": self.modified_sections,
            "has_changes": self.has_changes,
            "timestamp": self.timestamp.isoformat(),
        }

    def get_unified_diff(self) -> str:
        """Get unified diff format string."""
        old_lines = self.old_explanation.splitlines(keepends=True)
        new_lines = self.new_explanation.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="old_explanation",
            tofile="new_explanation",
        )
        return "".join(diff)

    def get_html_diff(self) -> str:
        """Get HTML formatted diff."""
        old_lines = self.old_explanation.splitlines()
        new_lines = self.new_explanation.splitlines()

        differ = difflib.HtmlDiff()
        return differ.make_table(
            old_lines,
            new_lines,
            fromdesc="Old Explanation",
            todesc="New Explanation",
        )


def _compute_similarity(text1: str, text2: str) -> float:
    """Compute similarity ratio between two texts.

    Args:
        text1: First text.
        text2: Second text.

    Returns:
        Similarity ratio (0.0 to 1.0).
    """
    return difflib.SequenceMatcher(None, text1, text2).ratio()


def _determine_significance(similarity: float) -> ChangeSignificance:
    """Determine significance based on similarity.

    Args:
        similarity: Similarity ratio.

    Returns:
        ChangeSignificance level.
    """
    if similarity >= 0.99:
        return ChangeSignificance.NONE
    elif similarity >= 0.85:
        return ChangeSignificance.MINOR
    elif similarity >= 0.60:
        return ChangeSignificance.MODERATE
    else:
        return ChangeSignificance.MAJOR


def _extract_diff_lines(old_text: str, new_text: str) -> List[DiffLine]:
    """Extract diff lines from two texts.

    Args:
        old_text: Original text.
        new_text: New text.

    Returns:
        List of DiffLine objects.
    """
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()

    diff_lines: List[DiffLine] = []
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)

    line_num = 1
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for idx in range(i2 - i1):
                diff_lines.append(
                    DiffLine(
                        line_number=line_num,
                        content=old_lines[i1 + idx],
                        diff_type=DiffType.UNCHANGED,
                        original_line=i1 + idx + 1,
                    )
                )
                line_num += 1
        elif tag == "delete":
            for idx in range(i2 - i1):
                diff_lines.append(
                    DiffLine(
                        line_number=line_num,
                        content=old_lines[i1 + idx],
                        diff_type=DiffType.REMOVED,
                        original_line=i1 + idx + 1,
                    )
                )
                line_num += 1
        elif tag == "insert":
            for idx in range(j2 - j1):
                diff_lines.append(
                    DiffLine(
                        line_number=line_num,
                        content=new_lines[j1 + idx],
                        diff_type=DiffType.ADDED,
                    )
                )
                line_num += 1
        elif tag == "replace":
            # First add removed lines
            for idx in range(i2 - i1):
                diff_lines.append(
                    DiffLine(
                        line_number=line_num,
                        content=old_lines[i1 + idx],
                        diff_type=DiffType.REMOVED,
                        original_line=i1 + idx + 1,
                    )
                )
                line_num += 1
            # Then add new lines
            for idx in range(j2 - j1):
                diff_lines.append(
                    DiffLine(
                        line_number=line_num,
                        content=new_lines[j1 + idx],
                        diff_type=DiffType.ADDED,
                    )
                )
                line_num += 1

    return diff_lines


def _extract_sections(diff_lines: List[DiffLine]) -> List[DiffSection]:
    """Extract contiguous change sections from diff lines.

    Args:
        diff_lines: List of diff lines.

    Returns:
        List of DiffSection objects.
    """
    sections: List[DiffSection] = []
    current_section: Optional[Dict[str, Any]] = None

    for line in diff_lines:
        if line.diff_type == DiffType.UNCHANGED:
            if current_section:
                sections.append(
                    DiffSection(
                        start_line=current_section["start"],
                        end_line=current_section["end"],
                        diff_type=DiffType.MODIFIED,
                        original_content="\n".join(current_section["removed"]),
                        new_content="\n".join(current_section["added"]),
                    )
                )
                current_section = None
        else:
            if not current_section:
                current_section = {
                    "start": line.line_number,
                    "end": line.line_number,
                    "removed": [],
                    "added": [],
                }
            current_section["end"] = line.line_number

            if line.diff_type == DiffType.REMOVED:
                current_section["removed"].append(line.content)
            elif line.diff_type == DiffType.ADDED:
                current_section["added"].append(line.content)

    # Handle last section
    if current_section:
        sections.append(
            DiffSection(
                start_line=current_section["start"],
                end_line=current_section["end"],
                diff_type=DiffType.MODIFIED,
                original_content="\n".join(current_section["removed"]),
                new_content="\n".join(current_section["added"]),
            )
        )

    return sections


def compare_explanations(
    old_result: ExplanationResult,
    new_result: ExplanationResult,
    code_changed: bool = False,
) -> ExplanationDiff:
    """Compare two explanation results.

    Args:
        old_result: Previous explanation result.
        new_result: New explanation result.
        code_changed: Whether the underlying code changed.

    Returns:
        ExplanationDiff with detailed comparison.
    """
    old_text = old_result.explanation
    new_text = new_result.explanation

    similarity = _compute_similarity(old_text, new_text)
    significance = _determine_significance(similarity)
    diff_lines = _extract_diff_lines(old_text, new_text)
    sections = _extract_sections(diff_lines)

    return ExplanationDiff(
        element_name=new_result.element.name,
        old_explanation=old_text,
        new_explanation=new_text,
        code_changed=code_changed,
        diff_lines=diff_lines,
        sections=sections,
        similarity_ratio=similarity,
        significance=significance,
        metadata={
            "old_confidence": old_result.confidence,
            "new_confidence": new_result.confidence,
            "confidence_diff": new_result.confidence - old_result.confidence,
        },
    )


def compare_explanation_texts(
    old_text: str,
    new_text: str,
    element_name: str = "unknown",
) -> ExplanationDiff:
    """Compare two explanation texts directly.

    Args:
        old_text: Previous explanation text.
        new_text: New explanation text.
        element_name: Name of the code element.

    Returns:
        ExplanationDiff with detailed comparison.
    """
    similarity = _compute_similarity(old_text, new_text)
    significance = _determine_significance(similarity)
    diff_lines = _extract_diff_lines(old_text, new_text)
    sections = _extract_sections(diff_lines)

    return ExplanationDiff(
        element_name=element_name,
        old_explanation=old_text,
        new_explanation=new_text,
        code_changed=False,
        diff_lines=diff_lines,
        sections=sections,
        similarity_ratio=similarity,
        significance=significance,
    )


class DiffTracker:
    """Tracks explanation diffs over time."""

    def __init__(self, max_diffs: int = 100) -> None:
        """Initialize diff tracker.

        Args:
            max_diffs: Maximum diffs to store per element.
        """
        self.max_diffs = max_diffs
        # element_name -> [diffs] (newest first)
        self._diffs: Dict[str, List[ExplanationDiff]] = {}

    def add_diff(self, diff: ExplanationDiff) -> None:
        """Add a diff to the tracker.

        Args:
            diff: ExplanationDiff to add.
        """
        name = diff.element_name
        if name not in self._diffs:
            self._diffs[name] = []

        self._diffs[name].insert(0, diff)

        # Trim old diffs
        if len(self._diffs[name]) > self.max_diffs:
            self._diffs[name] = self._diffs[name][: self.max_diffs]

    def get_history(
        self,
        element_name: str,
        limit: Optional[int] = None,
    ) -> List[ExplanationDiff]:
        """Get diff history for an element.

        Args:
            element_name: Name of code element.
            limit: Maximum diffs to return.

        Returns:
            List of diffs (newest first).
        """
        diffs = self._diffs.get(element_name, [])
        if limit:
            diffs = diffs[:limit]
        return diffs

    def get_significant_changes(
        self,
        element_name: str,
        min_significance: ChangeSignificance = ChangeSignificance.MODERATE,
    ) -> List[ExplanationDiff]:
        """Get only significant changes.

        Args:
            element_name: Name of code element.
            min_significance: Minimum significance level.

        Returns:
            List of significant diffs.
        """
        significance_order = [
            ChangeSignificance.NONE,
            ChangeSignificance.MINOR,
            ChangeSignificance.MODERATE,
            ChangeSignificance.MAJOR,
        ]
        min_idx = significance_order.index(min_significance)

        return [
            d
            for d in self._diffs.get(element_name, [])
            if significance_order.index(d.significance) >= min_idx
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics.

        Returns:
            Dictionary with statistics.
        """
        total_diffs = sum(len(diffs) for diffs in self._diffs.values())
        elements = len(self._diffs)

        return {
            "total_diffs": total_diffs,
            "tracked_elements": elements,
            "avg_diffs_per_element": total_diffs / elements if elements > 0 else 0,
        }


# Global diff tracker
_global_diff_tracker = DiffTracker()


def get_diff_tracker() -> DiffTracker:
    """Get the global diff tracker instance.

    Returns:
        DiffTracker singleton.
    """
    return _global_diff_tracker
