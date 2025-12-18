"""Explanation differ module for comparing and diffing explanations.

This module provides tools for comparing explanations, detecting changes,
and generating meaningful diffs between different versions of explanations.
"""

from __future__ import annotations

import difflib
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ExplanationDiffType(Enum):
    """Types of differences that can be detected."""

    ADDITION = auto()
    DELETION = auto()
    MODIFICATION = auto()
    MOVE = auto()
    UNCHANGED = auto()
    SEMANTIC = auto()


class DiffFormat(Enum):
    """Output formats for diff results."""

    UNIFIED = auto()
    SIDE_BY_SIDE = auto()
    HTML = auto()
    JSON = auto()
    CONTEXT = auto()
    INLINE = auto()


class DiffGranularity(Enum):
    """Granularity level for diff analysis."""

    CHARACTER = auto()
    WORD = auto()
    LINE = auto()
    PARAGRAPH = auto()
    SECTION = auto()


@dataclass
class DiffSegment:
    """A segment of a diff result."""

    content: str
    diff_type: ExplanationDiffType
    old_content: str | None = None
    start_line: int = 0
    end_line: int = 0
    start_char: int = 0
    end_char: int = 0
    similarity: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiffStats:
    """Statistics about a diff operation."""

    additions: int = 0
    deletions: int = 0
    modifications: int = 0
    moves: int = 0
    unchanged: int = 0
    total_old_lines: int = 0
    total_new_lines: int = 0
    similarity_ratio: float = 1.0

    @property
    def total_changes(self) -> int:
        """Get total number of changes."""
        return self.additions + self.deletions + self.modifications + self.moves

    @property
    def change_percentage(self) -> float:
        """Calculate change percentage."""
        total = max(self.total_old_lines, self.total_new_lines, 1)
        return (self.total_changes / total) * 100


@dataclass
class DiffExplanationVersion:
    """A versioned explanation for comparison."""

    content: str
    version: str = "1.0"
    timestamp: str | None = None
    author: str | None = None
    checksum: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate checksum if not provided."""
        if self.checksum is None:
            self.checksum = hashlib.md5(self.content.encode()).hexdigest()

    @property
    def lines(self) -> list[str]:
        """Get content as lines."""
        return self.content.splitlines()

    @property
    def words(self) -> list[str]:
        """Get content as words."""
        return self.content.split()

    @property
    def line_count(self) -> int:
        """Get number of lines."""
        return len(self.lines)

    @property
    def word_count(self) -> int:
        """Get number of words."""
        return len(self.words)


@dataclass
class DiffOptions:
    """Options for controlling diff behavior."""

    granularity: DiffGranularity = DiffGranularity.LINE
    format: DiffFormat = DiffFormat.UNIFIED
    context_lines: int = 3
    ignore_whitespace: bool = False
    ignore_case: bool = False
    ignore_blank_lines: bool = False
    detect_moves: bool = True
    semantic_diff: bool = False
    min_similarity: float = 0.6
    show_line_numbers: bool = True
    colorize: bool = True


@dataclass
class DiffResult:
    """Result of a diff operation."""

    segments: list[DiffSegment] = field(default_factory=list)
    stats: DiffStats = field(default_factory=DiffStats)
    old_version: DiffExplanationVersion | None = None
    new_version: DiffExplanationVersion | None = None
    format: DiffFormat = DiffFormat.UNIFIED
    options: DiffOptions = field(default_factory=DiffOptions)

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return self.stats.total_changes > 0

    @property
    def is_identical(self) -> bool:
        """Check if versions are identical."""
        return not self.has_changes

    def render(self) -> str:
        """Render the diff result."""
        renderer = DiffRenderer()
        return renderer.render(self)


class DiffAlgorithm(ABC):
    """Abstract base class for diff algorithms."""

    @abstractmethod
    def diff(
        self,
        old: DiffExplanationVersion,
        new: DiffExplanationVersion,
        options: DiffOptions,
    ) -> DiffResult:
        """Perform diff operation."""
        pass

    @abstractmethod
    def calculate_similarity(self, old: str, new: str) -> float:
        """Calculate similarity ratio between two strings."""
        pass


class LineDiffAlgorithm(DiffAlgorithm):
    """Line-based diff algorithm using difflib."""

    def diff(
        self,
        old: DiffExplanationVersion,
        new: DiffExplanationVersion,
        options: DiffOptions,
    ) -> DiffResult:
        """Perform line-based diff."""
        old_lines = self._preprocess(old.lines, options)
        new_lines = self._preprocess(new.lines, options)

        segments: list[DiffSegment] = []
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)

        additions = 0
        deletions = 0
        modifications = 0
        unchanged = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for idx in range(i1, i2):
                    segments.append(
                        DiffSegment(
                            content=old.lines[idx] if idx < len(old.lines) else "",
                            diff_type=ExplanationDiffType.UNCHANGED,
                            start_line=idx,
                            end_line=idx,
                        )
                    )
                unchanged += i2 - i1

            elif tag == "replace":
                # Check if it's a modification or semantic change
                old_block = "\n".join(old.lines[i1:i2])
                new_block = "\n".join(new.lines[j1:j2])
                similarity = self.calculate_similarity(old_block, new_block)

                for idx in range(i1, i2):
                    segments.append(
                        DiffSegment(
                            content=old.lines[idx] if idx < len(old.lines) else "",
                            diff_type=ExplanationDiffType.DELETION,
                            start_line=idx,
                            end_line=idx,
                        )
                    )

                for idx in range(j1, j2):
                    segments.append(
                        DiffSegment(
                            content=new.lines[idx] if idx < len(new.lines) else "",
                            diff_type=ExplanationDiffType.ADDITION,
                            old_content=old_block if similarity > options.min_similarity else None,
                            start_line=idx,
                            end_line=idx,
                            similarity=similarity,
                        )
                    )

                modifications += max(i2 - i1, j2 - j1)

            elif tag == "delete":
                for idx in range(i1, i2):
                    segments.append(
                        DiffSegment(
                            content=old.lines[idx] if idx < len(old.lines) else "",
                            diff_type=ExplanationDiffType.DELETION,
                            start_line=idx,
                            end_line=idx,
                        )
                    )
                deletions += i2 - i1

            elif tag == "insert":
                for idx in range(j1, j2):
                    segments.append(
                        DiffSegment(
                            content=new.lines[idx] if idx < len(new.lines) else "",
                            diff_type=ExplanationDiffType.ADDITION,
                            start_line=idx,
                            end_line=idx,
                        )
                    )
                additions += j2 - j1

        stats = DiffStats(
            additions=additions,
            deletions=deletions,
            modifications=modifications,
            unchanged=unchanged,
            total_old_lines=len(old.lines),
            total_new_lines=len(new.lines),
            similarity_ratio=matcher.ratio(),
        )

        return DiffResult(
            segments=segments,
            stats=stats,
            old_version=old,
            new_version=new,
            format=options.format,
            options=options,
        )

    def calculate_similarity(self, old: str, new: str) -> float:
        """Calculate similarity ratio."""
        return difflib.SequenceMatcher(None, old, new).ratio()

    def _preprocess(self, lines: list[str], options: DiffOptions) -> list[str]:
        """Preprocess lines based on options."""
        result = lines

        if options.ignore_whitespace:
            result = [line.strip() for line in result]

        if options.ignore_case:
            result = [line.lower() for line in result]

        if options.ignore_blank_lines:
            result = [line for line in result if line.strip()]

        return result


class WordDiffAlgorithm(DiffAlgorithm):
    """Word-based diff algorithm."""

    def diff(
        self,
        old: DiffExplanationVersion,
        new: DiffExplanationVersion,
        options: DiffOptions,
    ) -> DiffResult:
        """Perform word-based diff."""
        old_words = self._tokenize(old.content, options)
        new_words = self._tokenize(new.content, options)

        segments: list[DiffSegment] = []
        matcher = difflib.SequenceMatcher(None, old_words, new_words)

        additions = 0
        deletions = 0
        modifications = 0
        unchanged = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                content = " ".join(old_words[i1:i2])
                segments.append(
                    DiffSegment(
                        content=content,
                        diff_type=ExplanationDiffType.UNCHANGED,
                        start_char=i1,
                        end_char=i2,
                    )
                )
                unchanged += i2 - i1

            elif tag == "replace":
                old_content = " ".join(old_words[i1:i2])
                new_content = " ".join(new_words[j1:j2])
                similarity = self.calculate_similarity(old_content, new_content)

                segments.append(
                    DiffSegment(
                        content=old_content,
                        diff_type=ExplanationDiffType.DELETION,
                        start_char=i1,
                        end_char=i2,
                    )
                )
                segments.append(
                    DiffSegment(
                        content=new_content,
                        diff_type=ExplanationDiffType.ADDITION,
                        old_content=old_content,
                        start_char=j1,
                        end_char=j2,
                        similarity=similarity,
                    )
                )
                modifications += 1

            elif tag == "delete":
                content = " ".join(old_words[i1:i2])
                segments.append(
                    DiffSegment(
                        content=content,
                        diff_type=ExplanationDiffType.DELETION,
                        start_char=i1,
                        end_char=i2,
                    )
                )
                deletions += i2 - i1

            elif tag == "insert":
                content = " ".join(new_words[j1:j2])
                segments.append(
                    DiffSegment(
                        content=content,
                        diff_type=ExplanationDiffType.ADDITION,
                        start_char=j1,
                        end_char=j2,
                    )
                )
                additions += j2 - j1

        stats = DiffStats(
            additions=additions,
            deletions=deletions,
            modifications=modifications,
            unchanged=unchanged,
            total_old_lines=len(old_words),
            total_new_lines=len(new_words),
            similarity_ratio=matcher.ratio(),
        )

        return DiffResult(
            segments=segments,
            stats=stats,
            old_version=old,
            new_version=new,
            format=options.format,
            options=options,
        )

    def calculate_similarity(self, old: str, new: str) -> float:
        """Calculate similarity ratio."""
        return difflib.SequenceMatcher(None, old, new).ratio()

    def _tokenize(self, text: str, options: DiffOptions) -> list[str]:
        """Tokenize text into words."""
        import re

        words = re.findall(r"\S+", text)

        if options.ignore_case:
            words = [w.lower() for w in words]

        return words


class SemanticDiffAlgorithm(DiffAlgorithm):
    """Semantic diff algorithm for meaningful changes."""

    def diff(
        self,
        old: DiffExplanationVersion,
        new: DiffExplanationVersion,
        options: DiffOptions,
    ) -> DiffResult:
        """Perform semantic diff by sections."""
        old_sections = self._parse_sections(old.content)
        new_sections = self._parse_sections(new.content)

        segments: list[DiffSegment] = []
        additions = 0
        deletions = 0
        modifications = 0
        unchanged = 0

        # Match sections by header
        old_section_map = {s["header"]: s for s in old_sections}
        new_section_map = {s["header"]: s for s in new_sections}

        all_headers = set(old_section_map.keys()) | set(new_section_map.keys())

        for header in sorted(all_headers):
            old_section = old_section_map.get(header)
            new_section = new_section_map.get(header)

            if old_section and new_section:
                # Section exists in both
                if old_section["content"] == new_section["content"]:
                    segments.append(
                        DiffSegment(
                            content=new_section["content"],
                            diff_type=ExplanationDiffType.UNCHANGED,
                            metadata={"header": header},
                        )
                    )
                    unchanged += 1
                else:
                    similarity = self.calculate_similarity(
                        old_section["content"], new_section["content"]
                    )
                    segments.append(
                        DiffSegment(
                            content=new_section["content"],
                            diff_type=(
                                ExplanationDiffType.MODIFICATION
                                if similarity > options.min_similarity
                                else ExplanationDiffType.SEMANTIC
                            ),
                            old_content=old_section["content"],
                            similarity=similarity,
                            metadata={"header": header},
                        )
                    )
                    modifications += 1

            elif old_section:
                # Section removed
                segments.append(
                    DiffSegment(
                        content=old_section["content"],
                        diff_type=ExplanationDiffType.DELETION,
                        metadata={"header": header},
                    )
                )
                deletions += 1

            else:
                # Section added
                segments.append(
                    DiffSegment(
                        content=new_section["content"] if new_section else "",
                        diff_type=ExplanationDiffType.ADDITION,
                        metadata={"header": header},
                    )
                )
                additions += 1

        overall_similarity = self.calculate_similarity(old.content, new.content)

        stats = DiffStats(
            additions=additions,
            deletions=deletions,
            modifications=modifications,
            unchanged=unchanged,
            total_old_lines=len(old_sections),
            total_new_lines=len(new_sections),
            similarity_ratio=overall_similarity,
        )

        return DiffResult(
            segments=segments,
            stats=stats,
            old_version=old,
            new_version=new,
            format=options.format,
            options=options,
        )

    def calculate_similarity(self, old: str, new: str) -> float:
        """Calculate semantic similarity."""
        # Simple word overlap for now
        old_words = set(old.lower().split())
        new_words = set(new.lower().split())

        if not old_words and not new_words:
            return 1.0

        intersection = old_words & new_words
        union = old_words | new_words

        return len(intersection) / len(union) if union else 0.0

    def _parse_sections(self, content: str) -> list[dict[str, str]]:
        """Parse content into sections."""
        import re

        sections: list[dict[str, str]] = []
        current_header = "Introduction"
        current_content: list[str] = []

        for line in content.split("\n"):
            # Check for markdown headers
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                if current_content:
                    sections.append(
                        {
                            "header": current_header,
                            "content": "\n".join(current_content).strip(),
                        }
                    )
                current_header = header_match.group(2)
                current_content = []
            else:
                current_content.append(line)

        if current_content:
            sections.append(
                {
                    "header": current_header,
                    "content": "\n".join(current_content).strip(),
                }
            )

        return sections


class DiffRenderer:
    """Renderer for diff results."""

    def render(self, result: DiffResult) -> str:
        """Render diff result based on format."""
        if result.format == DiffFormat.UNIFIED:
            return self._render_unified(result)
        elif result.format == DiffFormat.SIDE_BY_SIDE:
            return self._render_side_by_side(result)
        elif result.format == DiffFormat.HTML:
            return self._render_html(result)
        elif result.format == DiffFormat.JSON:
            return self._render_json(result)
        elif result.format == DiffFormat.INLINE:
            return self._render_inline(result)
        return self._render_unified(result)

    def _render_unified(self, result: DiffResult) -> str:
        """Render unified diff format."""
        lines: list[str] = []

        # Header
        if result.old_version and result.new_version:
            lines.append(f"--- {result.old_version.version}")
            lines.append(f"+++ {result.new_version.version}")

        # Segments
        for segment in result.segments:
            if segment.diff_type == ExplanationDiffType.UNCHANGED:
                lines.append(f"  {segment.content}")
            elif segment.diff_type == ExplanationDiffType.DELETION:
                lines.append(f"- {segment.content}")
            elif segment.diff_type == ExplanationDiffType.ADDITION:
                lines.append(f"+ {segment.content}")
            elif segment.diff_type == ExplanationDiffType.MODIFICATION:
                lines.append(f"! {segment.content}")

        # Stats
        lines.append("")
        lines.append(
            f"Statistics: +{result.stats.additions} -{result.stats.deletions} ~{result.stats.modifications}"
        )
        lines.append(f"Similarity: {result.stats.similarity_ratio:.1%}")

        return "\n".join(lines)

    def _render_side_by_side(self, result: DiffResult) -> str:
        """Render side-by-side diff format."""
        lines: list[str] = []
        width = 40

        lines.append("=" * (width * 2 + 3))
        lines.append(f"{'Old'.center(width)} | {'New'.center(width)}")
        lines.append("=" * (width * 2 + 3))

        old_segments = [
            s
            for s in result.segments
            if s.diff_type in (ExplanationDiffType.DELETION, ExplanationDiffType.UNCHANGED)
        ]
        new_segments = [
            s
            for s in result.segments
            if s.diff_type in (ExplanationDiffType.ADDITION, ExplanationDiffType.UNCHANGED)
        ]

        max_len = max(len(old_segments), len(new_segments))

        for i in range(max_len):
            old_line = old_segments[i].content[:width] if i < len(old_segments) else ""
            new_line = new_segments[i].content[:width] if i < len(new_segments) else ""

            old_marker = (
                "-"
                if i < len(old_segments)
                and old_segments[i].diff_type == ExplanationDiffType.DELETION
                else " "
            )
            new_marker = (
                "+"
                if i < len(new_segments)
                and new_segments[i].diff_type == ExplanationDiffType.ADDITION
                else " "
            )

            lines.append(f"{old_marker}{old_line.ljust(width - 1)} | {new_marker}{new_line}")

        return "\n".join(lines)

    def _render_html(self, result: DiffResult) -> str:
        """Render HTML diff format."""
        import html as html_module

        parts: list[str] = []
        parts.append("<!DOCTYPE html><html><head><style>")
        parts.append(
            """
.diff { font-family: monospace; white-space: pre; }
.unchanged { color: #333; }
.addition { background: #dfd; color: #080; }
.deletion { background: #fdd; color: #800; text-decoration: line-through; }
.modification { background: #ffd; color: #880; }
.stats { margin-top: 20px; padding: 10px; background: #f5f5f5; }
"""
        )
        parts.append("</style></head><body>")
        parts.append('<div class="diff">')

        for segment in result.segments:
            class_name = segment.diff_type.name.lower()
            content = html_module.escape(segment.content)
            parts.append(f'<div class="{class_name}">{content}</div>')

        parts.append("</div>")
        parts.append('<div class="stats">')
        parts.append("<strong>Statistics:</strong> ")
        parts.append(f"+{result.stats.additions} additions, ")
        parts.append(f"-{result.stats.deletions} deletions, ")
        parts.append(f"~{result.stats.modifications} modifications<br>")
        parts.append(f"<strong>Similarity:</strong> {result.stats.similarity_ratio:.1%}")
        parts.append("</div>")
        parts.append("</body></html>")

        return "\n".join(parts)

    def _render_json(self, result: DiffResult) -> str:
        """Render JSON diff format."""
        import json

        data = {
            "stats": {
                "additions": result.stats.additions,
                "deletions": result.stats.deletions,
                "modifications": result.stats.modifications,
                "unchanged": result.stats.unchanged,
                "similarity_ratio": result.stats.similarity_ratio,
                "total_changes": result.stats.total_changes,
            },
            "segments": [
                {
                    "content": s.content,
                    "type": s.diff_type.name,
                    "similarity": s.similarity,
                    "old_content": s.old_content,
                }
                for s in result.segments
            ],
        }

        if result.old_version:
            data["old_version"] = {
                "version": result.old_version.version,
                "checksum": result.old_version.checksum,
            }

        if result.new_version:
            data["new_version"] = {
                "version": result.new_version.version,
                "checksum": result.new_version.checksum,
            }

        return json.dumps(data, indent=2)

    def _render_inline(self, result: DiffResult) -> str:
        """Render inline diff with markers."""
        parts: list[str] = []

        for segment in result.segments:
            if segment.diff_type == ExplanationDiffType.UNCHANGED:
                parts.append(segment.content)
            elif segment.diff_type == ExplanationDiffType.DELETION:
                parts.append(f"[-{segment.content}-]")
            elif segment.diff_type == ExplanationDiffType.ADDITION:
                parts.append(f"[+{segment.content}+]")
            elif segment.diff_type == ExplanationDiffType.MODIFICATION:
                parts.append(f"[~{segment.content}~]")

        return " ".join(parts)


class ExplanationDiffer:
    """Main class for diffing explanations."""

    def __init__(self) -> None:
        """Initialize with default algorithms."""
        self._algorithms: dict[DiffGranularity, DiffAlgorithm] = {
            DiffGranularity.LINE: LineDiffAlgorithm(),
            DiffGranularity.WORD: WordDiffAlgorithm(),
            DiffGranularity.SECTION: SemanticDiffAlgorithm(),
        }
        self._renderer = DiffRenderer()

    def register_algorithm(self, granularity: DiffGranularity, algorithm: DiffAlgorithm) -> None:
        """Register a diff algorithm."""
        self._algorithms[granularity] = algorithm

    def diff(
        self,
        old: str | DiffExplanationVersion,
        new: str | DiffExplanationVersion,
        options: DiffOptions | None = None,
    ) -> DiffResult:
        """Perform diff between two explanations."""
        if options is None:
            options = DiffOptions()

        if isinstance(old, str):
            old = DiffExplanationVersion(content=old, version="old")
        if isinstance(new, str):
            new = DiffExplanationVersion(content=new, version="new")

        algorithm = self._algorithms.get(options.granularity)
        if not algorithm:
            algorithm = self._algorithms[DiffGranularity.LINE]

        return algorithm.diff(old, new, options)

    def render(self, result: DiffResult) -> str:
        """Render a diff result."""
        return self._renderer.render(result)

    def quick_diff(self, old: str, new: str) -> str:
        """Quick diff with default options."""
        result = self.diff(old, new)
        return self.render(result)

    def html_diff(self, old: str, new: str) -> str:
        """Generate HTML diff."""
        options = DiffOptions(format=DiffFormat.HTML)
        result = self.diff(old, new, options)
        return self.render(result)

    def json_diff(self, old: str, new: str) -> str:
        """Generate JSON diff."""
        options = DiffOptions(format=DiffFormat.JSON)
        result = self.diff(old, new, options)
        return self.render(result)

    def side_by_side_diff(self, old: str, new: str) -> str:
        """Generate side-by-side diff."""
        options = DiffOptions(format=DiffFormat.SIDE_BY_SIDE)
        result = self.diff(old, new, options)
        return self.render(result)

    def word_diff(self, old: str, new: str) -> str:
        """Generate word-level diff."""
        options = DiffOptions(granularity=DiffGranularity.WORD, format=DiffFormat.INLINE)
        result = self.diff(old, new, options)
        return self.render(result)

    def semantic_diff(self, old: str, new: str) -> str:
        """Generate semantic section-based diff."""
        options = DiffOptions(granularity=DiffGranularity.SECTION, semantic_diff=True)
        result = self.diff(old, new, options)
        return self.render(result)


# Global instance
_explanation_differ: ExplanationDiffer | None = None


def get_explanation_differ() -> ExplanationDiffer:
    """Get or create global explanation differ."""
    global _explanation_differ
    if _explanation_differ is None:
        _explanation_differ = ExplanationDiffer()
    return _explanation_differ


def reset_explanation_differ() -> None:
    """Reset global explanation differ."""
    global _explanation_differ
    _explanation_differ = None


# Convenience functions
def diff_explanations(
    old: str,
    new: str,
    options: DiffOptions | None = None,
) -> DiffResult:
    """Diff two explanations."""
    return get_explanation_differ().diff(old, new, options)


def quick_diff(old: str, new: str) -> str:
    """Quick diff with defaults."""
    return get_explanation_differ().quick_diff(old, new)


def html_diff(old: str, new: str) -> str:
    """Generate HTML diff."""
    return get_explanation_differ().html_diff(old, new)


def json_diff(old: str, new: str) -> str:
    """Generate JSON diff."""
    return get_explanation_differ().json_diff(old, new)


def side_by_side_diff(old: str, new: str) -> str:
    """Generate side-by-side diff."""
    return get_explanation_differ().side_by_side_diff(old, new)


def word_diff(old: str, new: str) -> str:
    """Generate word-level diff."""
    return get_explanation_differ().word_diff(old, new)


def semantic_diff(old: str, new: str) -> str:
    """Generate semantic diff."""
    return get_explanation_differ().semantic_diff(old, new)


def create_version(
    content: str,
    version: str = "1.0",
    author: str | None = None,
    **kwargs: Any,
) -> DiffExplanationVersion:
    """Create an explanation version."""
    return DiffExplanationVersion(content=content, version=version, author=author, **kwargs)


def calculate_similarity(old: str, new: str) -> float:
    """Calculate similarity between two texts."""
    return difflib.SequenceMatcher(None, old, new).ratio()


def get_diff_stats(old: str, new: str) -> DiffStats:
    """Get diff statistics."""
    result = diff_explanations(old, new)
    return result.stats
