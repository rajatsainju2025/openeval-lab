"""
String building optimizations using IO buffers.

Replaces inefficient string concatenation with O(1) StringIO operations.
"""

from io import StringIO
from typing import List, Optional


def build_table(
    headers: List[str],
    rows: List[List[str]],
    column_widths: Optional[List[int]] = None,
) -> str:
    """Build formatted table efficiently using StringIO.

    Args:
        headers: Column headers
        rows: Table rows (list of cells per row)
        column_widths: Optional custom column widths

    Returns:
        Formatted table string
    """
    buf = StringIO()

    # Calculate column widths
    if not column_widths:
        column_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(column_widths):
                    column_widths[i] = max(column_widths[i], len(str(cell)))

    # Write header
    header_line = " | ".join(str(h).ljust(w) for h, w in zip(headers, column_widths))
    buf.write(header_line)
    buf.write("\n")
    buf.write("-" * len(header_line))
    buf.write("\n")

    # Write rows
    for row in rows:
        row_line = " | ".join(str(cell).ljust(w) for cell, w in zip(row, column_widths))
        buf.write(row_line)
        buf.write("\n")

    return buf.getvalue()


def build_report(sections: List[tuple[str, str]]) -> str:
    """Build multi-section report efficiently using StringIO.

    Args:
        sections: List of (title, content) tuples

    Returns:
        Formatted report string
    """
    buf = StringIO()

    for i, (title, content) in enumerate(sections):
        if i > 0:
            buf.write("\n")
        buf.write(f"## {title}\n\n")
        buf.write(content)
        buf.write("\n")

    return buf.getvalue()


def join_lines(lines: List[str], separator: str = "\n") -> str:
    """Join lines efficiently using StringIO.

    Args:
        lines: List of lines to join
        separator: Line separator (default: newline)

    Returns:
        Joined string
    """
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]

    buf = StringIO()
    for i, line in enumerate(lines):
        buf.write(line)
        if i < len(lines) - 1:
            buf.write(separator)
    return buf.getvalue()


def format_metrics(metrics: dict, precision: int = 4) -> str:
    """Format metrics dictionary efficiently using StringIO.

    Args:
        metrics: Metrics dictionary
        precision: Decimal precision for floats

    Returns:
        Formatted metrics string
    """
    buf = StringIO()

    for key, value in metrics.items():
        if isinstance(value, float):
            formatted_value = f"{value:.{precision}f}"
        else:
            formatted_value = str(value)
        buf.write(f"{key}: {formatted_value}\n")

    return buf.getvalue()


class EfficientStringBuilder:
    """Reusable string builder for efficient concatenation."""

    def __init__(self):
        """Initialize string builder."""
        self._buffer = StringIO()

    def append(self, text: str) -> "EfficientStringBuilder":
        """Append text to buffer.

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
            text: Text to append (default: empty line)

        Returns:
            Self for method chaining
        """
        self._buffer.write(text)
        self._buffer.write("\n")
        return self

    def append_formatted(self, template: str, **kwargs) -> "EfficientStringBuilder":
        """Append formatted template.

        Args:
            template: Format string
            **kwargs: Format arguments

        Returns:
            Self for method chaining
        """
        self._buffer.write(template.format(**kwargs))
        return self

    def get(self) -> str:
        """Get final string."""
        return self._buffer.getvalue()

    def clear(self) -> None:
        """Clear buffer."""
        self._buffer = StringIO()

    def __str__(self) -> str:
        """Convert to string."""
        return self.get()
