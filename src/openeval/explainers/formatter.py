"""Code formatting and presentation utilities.

Provides syntax highlighting, annotation, and multi-format output.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .types import ExplanationResult


class OutputFormat(str, Enum):
    """Output format options."""

    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    ANSI = "ansi"


@dataclass
class LineAnnotation:
    """Annotation for a code line."""

    line_number: int
    annotation: str
    level: str = "info"  # info, warning, error, highlight


class CodeFormatter:
    """Format code with syntax highlighting and annotations."""

    ANSI_COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "cyan": "\033[36m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "red": "\033[31m",
        "blue": "\033[34m",
    }

    MARKDOWN_LANGUAGE = "python"

    def __init__(self) -> None:
        """Initialize the formatter."""
        pass

    def format_code_block(
        self,
        code: str,
        format: OutputFormat = OutputFormat.TEXT,
        line_numbers: bool = True,
        max_lines: Optional[int] = None,
    ) -> str:
        """Format code block for display.

        Args:
            code: Source code to format.
            format: Output format.
            line_numbers: Whether to include line numbers.
            max_lines: Maximum lines to show, None for unlimited.

        Returns:
            Formatted code string.
        """
        lines = code.split("\n")

        if max_lines and len(lines) > max_lines:
            lines = lines[:max_lines]
            lines.append(f"... ({len(code.split(chr(10))) - max_lines} more lines)")

        if format == OutputFormat.MARKDOWN:
            return self._format_markdown(lines, line_numbers)
        elif format == OutputFormat.ANSI:
            return self._format_ansi(lines, line_numbers)
        elif format == OutputFormat.HTML:
            return self._format_html(lines, line_numbers)
        else:  # TEXT
            return self._format_text(lines, line_numbers)

    def format_with_annotations(
        self,
        code: str,
        annotations: List[LineAnnotation],
        format: OutputFormat = OutputFormat.TEXT,
    ) -> str:
        """Format code with line annotations.

        Args:
            code: Source code.
            annotations: List of line annotations.
            format: Output format.

        Returns:
            Formatted code with annotations.
        """
        lines = code.split("\n")
        formatted = self.format_code_block(code, format, line_numbers=True)

        if format == OutputFormat.MARKDOWN:
            return self._add_annotations_markdown(formatted, annotations)
        elif format == OutputFormat.ANSI:
            return self._add_annotations_ansi(lines, annotations)
        else:
            return self._add_annotations_text(formatted, annotations)

    def format_explanation_result(
        self,
        result: ExplanationResult,
        format: OutputFormat = OutputFormat.TEXT,
        include_code: bool = True,
    ) -> str:
        """Format an explanation result for display.

        Args:
            result: ExplanationResult to format.
            format: Output format.
            include_code: Whether to include the code snippet.

        Returns:
            Formatted explanation.
        """
        parts = []

        # Header
        parts.append(self._format_header(result, format))

        # Code snippet
        if include_code:
            code_formatted = self.format_code_block(
                result.element.source_code, format, line_numbers=True, max_lines=20
            )
            parts.append(self._format_section("Code", code_formatted, format))

        # Explanation
        parts.append(self._format_section("Explanation", result.explanation, format))

        # Metadata
        if result.analysis_metadata:
            metadata_str = self._format_metadata(result.analysis_metadata, format)
            parts.append(self._format_section("Analysis", metadata_str, format))

        return self._join_sections(parts, format)

    def _format_text(self, lines: List[str], line_numbers: bool) -> str:
        """Format as plain text with optional line numbers."""
        formatted_lines = []
        for i, line in enumerate(lines, 1):
            if line_numbers:
                formatted_lines.append(f"{i:4d} | {line}")
            else:
                formatted_lines.append(line)
        return "\n".join(formatted_lines)

    def _format_markdown(self, lines: List[str], line_numbers: bool) -> str:
        """Format as markdown code block."""
        code = "\n".join(lines)
        return f"```{self.MARKDOWN_LANGUAGE}\n{code}\n```"

    def _format_ansi(self, lines: List[str], line_numbers: bool) -> str:
        """Format with ANSI color codes."""
        formatted_lines = []
        for i, line in enumerate(lines, 1):
            # Color line numbers
            if line_numbers:
                num_str = f"{self.ANSI_COLORS['dim']}{i:4d}{self.ANSI_COLORS['reset']}"
                colored_line = f"{num_str} | {line}"
            else:
                colored_line = line

            formatted_lines.append(colored_line)

        return "\n".join(formatted_lines)

    def _format_html(self, lines: List[str], line_numbers: bool) -> str:
        """Format as HTML."""
        html_lines = ['<pre><code class="language-python">']

        for i, line in enumerate(lines, 1):
            # Escape HTML
            line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            if line_numbers:
                html_lines.append(f'<span class="line-num">{i:4d}</span> | {line}')
            else:
                html_lines.append(line)

        html_lines.append("</code></pre>")
        return "\n".join(html_lines)

    def _format_header(self, result: ExplanationResult, format: OutputFormat) -> str:
        """Format result header."""
        header_parts = [
            f"Element: {result.element.name}",
            f"Type: {result.element.type.value}",
            f"Lines: {result.element.line_start}-{result.element.line_end}",
            f"Confidence: {result.confidence:.1%}",
        ]

        if format == OutputFormat.MARKDOWN:
            return "## " + "\n- ".join([header_parts[0]] + header_parts[1:])
        elif format == OutputFormat.ANSI:
            return "\n".join(
                [f"{self.ANSI_COLORS['bold']}{p}{self.ANSI_COLORS['reset']}" for p in header_parts]
            )
        else:
            return "\n".join(header_parts)

    def _format_section(self, title: str, content: str, format: OutputFormat) -> str:
        """Format a section with title and content."""
        if format == OutputFormat.MARKDOWN:
            return f"### {title}\n\n{content}"
        elif format == OutputFormat.ANSI:
            colored_title = (
                f"{self.ANSI_COLORS['bold']}{self.ANSI_COLORS['cyan']}"
                f"{title}{self.ANSI_COLORS['reset']}"
            )
            return f"{colored_title}\n{content}"
        else:
            return f"{title}:\n{content}"

    def _format_metadata(self, metadata: dict, format: OutputFormat) -> str:
        """Format metadata dictionary."""
        lines = []
        for key, value in metadata.items():
            if format == OutputFormat.MARKDOWN:
                lines.append(f"- **{key}**: {value}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def _add_annotations_markdown(
        self, formatted_code: str, annotations: List[LineAnnotation]
    ) -> str:
        """Add annotations to markdown-formatted code."""
        if not annotations:
            return formatted_code

        result = [formatted_code]
        result.append("\n#### Notes:")

        for ann in annotations:
            result.append(f"- **Line {ann.line_number}** ({ann.level}): {ann.annotation}")

        return "\n".join(result)

    def _add_annotations_ansi(self, lines: List[str], annotations: List[LineAnnotation]) -> str:
        """Add annotations to ANSI-formatted code."""
        if not annotations:
            return self._format_ansi(lines, True)

        ann_map = {a.line_number: a for a in annotations}
        result = []

        for i, line in enumerate(lines, 1):
            result.append(f"{i:4d} | {line}")
            if i in ann_map:
                ann = ann_map[i]
                color = {
                    "error": self.ANSI_COLORS["red"],
                    "warning": self.ANSI_COLORS["yellow"],
                    "highlight": self.ANSI_COLORS["green"],
                    "info": self.ANSI_COLORS["blue"],
                }.get(ann.level, self.ANSI_COLORS["dim"])

                result.append(f"     {color}↳ {ann.annotation}{self.ANSI_COLORS['reset']}")

        return "\n".join(result)

    def _add_annotations_text(self, formatted_code: str, annotations: List[LineAnnotation]) -> str:
        """Add annotations to plain text code."""
        if not annotations:
            return formatted_code

        result = [formatted_code]
        result.append("\nNotes:")

        for ann in annotations:
            result.append(f"  Line {ann.line_number} ({ann.level}): {ann.annotation}")

        return "\n".join(result)

    def _join_sections(self, parts: List[str], format: OutputFormat) -> str:
        """Join formatted sections."""
        if format == OutputFormat.MARKDOWN:
            return "\n\n".join(parts)
        elif format == OutputFormat.ANSI:
            return "\n\n".join(parts)
        else:
            return "\n\n".join(parts)
