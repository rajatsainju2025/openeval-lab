"""
Consolidated String Utilities Module

Merges string_utils.py and string_utilities.py into a single efficient module.
Provides efficient string building, formatting, and text processing utilities
for the entire OpenEval Lab codebase.

Key optimizations:
- StringIO-based builder avoids O(n²) string concatenation
- Minimal memory overhead through efficient buffering
- Unified interface for all string operations across codebase
"""

from io import StringIO
from typing import List, Dict, Any, Optional, Iterable, Sequence
import json


class EfficientStringBuilder:
    """Efficient string builder using StringIO to avoid O(n²) concatenation.

    Provides method chaining for fluent API usage.
    """

    def __init__(self, initial: str = ""):
        """Initialize with optional initial content."""
        self.buffer = StringIO()
        if initial:
            self.buffer.write(initial)

    def append(self, text: str) -> "EfficientStringBuilder":
        """Append text to the builder.

        Args:
            text: Text to append

        Returns:
            Self for method chaining
        """
        self.buffer.write(text)
        return self

    def append_line(self, text: str = "") -> "EfficientStringBuilder":
        """Append text followed by newline.

        Args:
            text: Text to append (empty string for blank line)

        Returns:
            Self for method chaining
        """
        self.buffer.write(text)
        self.buffer.write("\n")
        return self

    def append_lines(self, lines: Iterable[str]) -> "EfficientStringBuilder":
        """Append multiple lines.

        Args:
            lines: Iterable of lines

        Returns:
            Self for method chaining
        """
        for line in lines:
            self.append_line(line)
        return self

    def prepend(self, text: str) -> "EfficientStringBuilder":
        """Prepend text (less efficient, use sparingly).

        Args:
            text: Text to prepend

        Returns:
            Self for method chaining
        """
        current = self.buffer.getvalue()
        self.buffer = StringIO()
        self.buffer.write(text)
        self.buffer.write(current)
        return self

    def clear(self) -> "EfficientStringBuilder":
        """Clear all content.

        Returns:
            Self for method chaining
        """
        self.buffer = StringIO()
        return self

    def get(self) -> str:
        """Get the built string without consuming it."""
        return self.buffer.getvalue()

    def build(self) -> str:
        """Get the final string."""
        return self.buffer.getvalue()

    def __str__(self) -> str:
        """Return built string when converted to string."""
        return self.get()

    def __len__(self) -> int:
        """Return length of built string."""
        return len(self.get())

    def __repr__(self) -> str:
        """Return representation."""
        content = self.get()
        if len(content) > 50:
            return f"EfficientStringBuilder({content[:50]}...)"
        return f"EfficientStringBuilder({content})"


def build_table(
    headers: List[str],
    rows: List[List[Any]],
    column_widths: Optional[List[int]] = None,
    align: str = "left",
) -> str:
    """Build a formatted table as a string.

    Args:
        headers: Column headers
        rows: Rows of data
        column_widths: Optional column widths (auto-calculated if None)
        align: Text alignment ('left', 'right', 'center')

    Returns:
        Formatted table string
    """
    builder = EfficientStringBuilder()

    if not headers or not rows:
        return ""

    # Calculate column widths
    if column_widths is None:
        column_widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                column_widths[i] = max(column_widths[i], len(str(cell)))

    # Build header separator
    separator = "+" + "+".join("-" * (w + 2) for w in column_widths) + "+"
    builder.append_line(separator)

    # Build header row
    header_cells = []
    for i, header in enumerate(headers):
        width = column_widths[i]
        if align == "right":
            cell = str(header).rjust(width)
        elif align == "center":
            cell = str(header).center(width)
        else:
            cell = str(header).ljust(width)
        header_cells.append(f" {cell} ")

    builder.append_line("|" + "|".join(header_cells) + "|")
    builder.append_line(separator)

    # Build data rows
    for row in rows:
        row_cells = []
        for i, cell in enumerate(row):
            width = column_widths[i]
            cell_str = str(cell)
            if align == "right":
                formatted = cell_str.rjust(width)
            elif align == "center":
                formatted = cell_str.center(width)
            else:
                formatted = cell_str.ljust(width)
            row_cells.append(f" {formatted} ")

        builder.append_line("|" + "|".join(row_cells) + "|")

    builder.append_line(separator)
    return builder.get()


def build_report(
    title: str,
    sections: Dict[str, Any],
    include_summary: bool = True,
) -> str:
    """Build a formatted report.

    Args:
        title: Report title
        sections: Dict of section_name -> section_content
        include_summary: Whether to include summary section

    Returns:
        Formatted report string
    """
    builder = EfficientStringBuilder()

    # Title
    builder.append_line("=" * 70)
    builder.append_line(title.center(70))
    builder.append_line("=" * 70)
    builder.append_line()

    # Sections
    for section_name, section_content in sections.items():
        builder.append_line(f"## {section_name}")
        builder.append_line()

        if isinstance(section_content, dict):
            for key, value in section_content.items():
                builder.append_line(f"  {key}: {value}")
        elif isinstance(section_content, list):
            for item in section_content:
                builder.append_line(f"  - {item}")
        else:
            builder.append_line(str(section_content))

        builder.append_line()

    # Summary
    if include_summary:
        builder.append_line("=" * 70)
        builder.append_line(f"Total sections: {len(sections)}")
        builder.append_line("=" * 70)

    return builder.get()


def join_lines(
    items: Iterable[str],
    separator: str = "\n",
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Join items with separator and optional prefix/suffix.

    Args:
        items: Items to join
        separator: Separator between items
        prefix: Prefix for entire result
        suffix: Suffix for entire result

    Returns:
        Joined string
    """
    content = separator.join(str(item) for item in items)
    return f"{prefix}{content}{suffix}"


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format metrics dictionary as readable string.

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
    """Truncate string to maximum length with suffix.

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


def indent_lines(text: str, indent: int = 2, char: str = " ") -> str:
    """Indent all lines in text.

    Args:
        text: Text to indent
        indent: Number of indent characters
        char: Indent character (space or tab)

    Returns:
        Indented text
    """
    indent_str = char * indent
    lines = text.split("\n")
    return "\n".join(indent_str + line for line in lines)


def wrap_text(text: str, width: int = 80, prefix: str = "") -> str:
    """Wrap text to specified width.

    Args:
        text: Text to wrap
        width: Maximum line width
        prefix: Prefix for wrapped lines

    Returns:
        Wrapped text
    """
    words = text.split()
    lines = []
    current_line = prefix

    for word in words:
        if len(current_line) + len(word) + 1 <= width:
            if len(current_line) > len(prefix):
                current_line += " " + word
            else:
                current_line += word
        else:
            lines.append(current_line)
            current_line = prefix + word

    if current_line.strip():
        lines.append(current_line)

    return "\n".join(lines)


def format_json(data: Any, pretty: bool = True, indent: int = 2) -> str:
    """Format data as JSON.

    Args:
        data: Data to format
        pretty: Whether to pretty-print
        indent: Indentation level if pretty

    Returns:
        JSON string
    """
    if pretty:
        return json.dumps(data, indent=indent, default=str)
    else:
        return json.dumps(data, default=str)


def parse_json_lines(text: str) -> List[Dict[str, Any]]:
    """Parse JSONL (JSON Lines) format.

    Args:
        text: JSONL text

    Returns:
        List of parsed dictionaries
    """
    results = []
    for line in text.strip().split("\n"):
        if line.strip():
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def to_jsonl(items: Sequence[Dict[str, Any]]) -> str:
    """Convert items to JSONL format.

    Args:
        items: Items to convert

    Returns:
        JSONL string
    """
    builder = EfficientStringBuilder()
    for item in items:
        builder.append_line(json.dumps(item, default=str))
    return builder.get()


def sanitize_filename(filename: str) -> str:
    """Sanitize string for use as filename.

    Args:
        filename: Filename to sanitize

    Returns:
        Sanitized filename
    """
    import re

    # Replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Replace multiple underscores with single
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores and spaces
    sanitized = sanitized.strip("_ ")
    return sanitized


def highlight_text(text: str, width: int = 80, char: str = "=") -> str:
    """Highlight text with border characters.

    Args:
        text: Text to highlight
        width: Width of highlight
        char: Character to use for border

    Returns:
        Highlighted text
    """
    border = char * width
    padding = " " * ((width - len(text)) // 2)
    return f"{border}\n{padding}{text}\n{border}"


def compare_strings(str1: str, str2: str) -> Dict[str, Any]:
    """Compare two strings and return differences.

    Args:
        str1: First string
        str2: Second string

    Returns:
        Comparison dictionary
    """
    return {
        "str1": str1[:100],
        "str2": str2[:100],
        "equal": str1 == str2,
        "len_str1": len(str1),
        "len_str2": len(str2),
        "first_diff_pos": next((i for i, (c1, c2) in enumerate(zip(str1, str2)) if c1 != c2), -1),
    }
