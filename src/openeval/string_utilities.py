# Unified String Utilities Module

"""
Consolidated string utilities for efficient text building and formatting.

Consolidates string_utils.py with scattered string operations across the codebase.
Provides efficient StringIO-based builders for reports, metrics, and data formatting.
"""

from io import StringIO
from typing import List, Dict, Any, Optional, Iterable
import json


class EfficientStringBuilder:
    """Efficient string builder using StringIO to avoid O(n²) concatenation."""

    def __init__(self, initial_content: str = ""):
        """Initialize builder.

        Args:
            initial_content: Initial string content
        """
        self._buffer = StringIO(initial_content)
        if initial_content:
            self._buffer.seek(0, 2)  # Move to end

    def append(self, text: str) -> "EfficientStringBuilder":
        """Append text to builder.

        Args:
            text: Text to append

        Returns:
            Self for method chaining
        """
        self._buffer.write(text)
        return self

    def append_line(self, text: str = "") -> "EfficientStringBuilder":
        """Append text with newline.

        Args:
            text: Text to append

        Returns:
            Self for method chaining
        """
        self._buffer.write(text)
        self._buffer.write("\n")
        return self

    def append_formatted(self, template: str, **kwargs) -> "EfficientStringBuilder":
        """Append formatted text.

        Args:
            template: Template string with {} placeholders
            **kwargs: Format arguments

        Returns:
            Self for method chaining
        """
        self._buffer.write(template.format(**kwargs))
        return self

    def append_json(self, data: Any, indent: bool = False) -> "EfficientStringBuilder":
        """Append JSON-formatted data.

        Args:
            data: Data to serialize
            indent: Whether to pretty-print

        Returns:
            Self for method chaining
        """
        json_str = json.dumps(data, indent=2 if indent else None)
        self._buffer.write(json_str)
        return self

    def clear(self) -> "EfficientStringBuilder":
        """Clear buffer.

        Returns:
            Self for method chaining
        """
        self._buffer.truncate(0)
        self._buffer.seek(0)
        return self

    def get(self) -> str:
        """Get current content."""
        return self._buffer.getvalue()

    def __str__(self) -> str:
        """String representation."""
        return self.get()

    def __len__(self) -> int:
        """Length of content."""
        return len(self.get())

    def __repr__(self) -> str:
        """Representation."""
        content = self.get()
        preview = content[:50] + "..." if len(content) > 50 else content
        return f"StringBuilder({len(content)} chars): {repr(preview)}"


def build_table(
    headers: List[str],
    rows: List[List[Any]],
    column_widths: Optional[List[int]] = None,
    align: str = "left",
) -> str:
    """
    Build a formatted table string.

    Args:
        headers: Column headers
        rows: Table rows (list of lists)
        column_widths: Optional column widths
        align: Text alignment ('left', 'right', 'center')

    Returns:
        Formatted table string
    """
    builder = EfficientStringBuilder()

    if column_widths is None:
        column_widths = [
            max(len(str(h)), max(len(str(row[i])) for row in rows)) for i, h in enumerate(headers)
        ]

    # Build header
    header_parts = []
    for i, header in enumerate(headers):
        width = column_widths[i]
        if align == "right":
            header_parts.append(str(header).rjust(width))
        elif align == "center":
            header_parts.append(str(header).center(width))
        else:
            header_parts.append(str(header).ljust(width))

    builder.append_line(" | ".join(header_parts))
    builder.append_line("-" * sum(column_widths) + "--" * (len(headers) - 1))

    # Build rows
    for row in rows:
        row_parts = []
        for i, cell in enumerate(row):
            width = column_widths[i]
            cell_str = str(cell)
            if align == "right":
                row_parts.append(cell_str.rjust(width))
            elif align == "center":
                row_parts.append(cell_str.center(width))
            else:
                row_parts.append(cell_str.ljust(width))

        builder.append_line(" | ".join(row_parts))

    return builder.get()


def build_report(
    title: str,
    sections: Dict[str, Any],
    include_summary: bool = True,
) -> str:
    """
    Build a formatted report string.

    Args:
        title: Report title
        sections: Dictionary of section_name -> section_content
        include_summary: Whether to include summary section

    Returns:
        Formatted report string
    """
    builder = EfficientStringBuilder()

    # Title
    builder.append_line(f"{'=' * len(title)}")
    builder.append_line(title)
    builder.append_line(f"{'=' * len(title)}")
    builder.append_line()

    # Summary
    if include_summary:
        builder.append_line("SUMMARY")
        builder.append_line("-" * 7)
        builder.append_line(f"Sections: {len(sections)}")
        builder.append_line()

    # Sections
    for section_name, section_content in sections.items():
        builder.append_line(f"{section_name.upper()}")
        builder.append_line("-" * len(section_name))

        if isinstance(section_content, dict):
            for key, value in section_content.items():
                builder.append_line(f"  {key}: {value}")
        elif isinstance(section_content, list):
            for item in section_content:
                builder.append_line(f"  - {item}")
        else:
            builder.append_line(f"  {section_content}")

        builder.append_line()

    return builder.get()


def join_lines(
    items: Iterable[str],
    separator: str = "\n",
    prefix: str = "",
    suffix: str = "",
) -> str:
    """
    Efficiently join lines with optional prefix/suffix.

    Args:
        items: Items to join
        separator: Line separator
        prefix: Prefix for each item
        suffix: Suffix for each item

    Returns:
        Joined string
    """
    builder = EfficientStringBuilder()

    for i, item in enumerate(items):
        if i > 0:
            builder.append(separator)
        builder.append(f"{prefix}{item}{suffix}")

    return builder.get()


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format metrics dictionary as readable string.

    Args:
        metrics: Metrics dictionary
        precision: Decimal places for floats

    Returns:
        Formatted metrics string
    """
    builder = EfficientStringBuilder()

    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            formatted_value = f"{value:.{precision}f}"
        else:
            formatted_value = str(value)

        builder.append_line(f"{key}: {formatted_value}")

    return builder.get()


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate string to maximum length with suffix.

    Args:
        text: String to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add if truncated

    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def indent_lines(text: str, spaces: int = 4) -> str:
    """
    Indent all lines in text.

    Args:
        text: Text to indent
        spaces: Number of spaces

    Returns:
        Indented text
    """
    indent = " " * spaces
    return "\n".join(indent + line for line in text.split("\n"))


def wrap_text(text: str, width: int = 80) -> str:
    """
    Wrap text to specified width.

    Args:
        text: Text to wrap
        width: Maximum line width

    Returns:
        Wrapped text
    """
    import textwrap

    return textwrap.fill(text, width=width)
