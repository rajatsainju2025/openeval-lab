"""
Unified String Utilities for OpenEval Lab.

High-performance string operations including:
- Efficient string building with StringIO (5-10x faster than concatenation)
- Pre-compiled regex patterns for reuse (3-5x faster)
- Optimized text normalization and tokenization with LRU caching
- Batch string processing operations
- Table, report, and metrics formatting utilities

This module consolidates:
- string_utils.py (original)
- string_utilities.py (deprecated wrapper)
- string_utils_consolidated.py (partial merge)
- string_utils_optimized.py (performance features)
"""

from __future__ import annotations

import json
import re
import string
import unicodedata
from functools import lru_cache
from io import StringIO
from typing import Any, Dict, Iterable, List, Optional, Pattern, Sequence

__all__ = [
    # Core classes
    "EfficientStringBuilder",
    # Table/report building
    "build_table",
    "build_report",
    "join_lines",
    # Metrics formatting
    "format_metrics",
    "format_size",
    # Text processing
    "normalize_text",
    "clean_text",
    "extract_tokens",
    "batch_normalize_text",
    "tokenize",
    "strip_punctuation",
    # String utilities
    "truncate_text",
    "indent_lines",
    "wrap_text",
    # JSON utilities
    "format_json",
    "parse_jsonl",
    "to_jsonl",
    # Misc utilities
    "sanitize_filename",
    "highlight_text",
    "compare_strings",
    "get_pattern",
]

# =============================================================================
# Pre-compiled Regex Patterns (3-5x faster than on-demand compilation)
# =============================================================================

_COMPILED_PATTERNS: Dict[str, Pattern[str]] = {
    "html_tags": re.compile(r"<[^>]+>"),
    "whitespace": re.compile(r"\s+"),
    "punctuation": re.compile(r"[^\w\s]"),
    "numbers": re.compile(r"\b\d+\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "url": re.compile(r"https?://[^\s<>\"{}|\\^`[\]]+"),
    "unicode_quotes": re.compile(r"[" "''„‚«»‹›「」『』〝〞〟]"),
    "unicode_spaces": re.compile(r"[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000\uFEFF]"),
    "invalid_filename": re.compile(r'[<>:"/\\|?*]'),
    "multiple_underscores": re.compile(r"_+"),
}

# Character translation tables for fast O(n) character replacement
_PUNCTUATION_TRANSLATOR = str.maketrans(string.punctuation, " " * len(string.punctuation))
_NORMALIZE_TRANSLATOR = str.maketrans(
    {
        "\u2018": "'",  # Left single quotation mark
        "\u2019": "'",  # Right single quotation mark
        "\u201c": '"',  # Left double quotation mark
        "\u201d": '"',  # Right double quotation mark
        "\u2013": "-",  # En dash
        "\u2014": "-",  # Em dash
        "\u2026": "...",  # Horizontal ellipsis
    }
)


def get_pattern(name: str) -> Optional[Pattern[str]]:
    """Get pre-compiled regex pattern by name.

    Available patterns: html_tags, whitespace, punctuation, numbers,
    email, url, unicode_quotes, unicode_spaces

    Args:
        name: Pattern name

    Returns:
        Compiled regex pattern or None if not found
    """
    return _COMPILED_PATTERNS.get(name)


# =============================================================================
# Efficient String Builder
# =============================================================================


class EfficientStringBuilder:
    """High-performance string builder using StringIO.

    Provides 5-10x performance improvement over string concatenation
    for building large strings incrementally. Uses fluent API for
    method chaining.

    Example:
        >>> builder = EfficientStringBuilder()
        >>> builder.append("Hello").append_line(" World").append_line("!")
        >>> print(builder.get())
        Hello World
        !
    """

    __slots__ = ("_buffer", "_length")

    def __init__(self, initial: str = "") -> None:
        """Initialize builder with optional initial content.

        Args:
            initial: Optional initial string content
        """
        self._buffer = StringIO()
        self._length = 0
        if initial:
            self._buffer.write(initial)
            self._length = len(initial)

    def append(self, text: str) -> EfficientStringBuilder:
        """Append text without newline.

        Args:
            text: Text to append

        Returns:
            Self for method chaining
        """
        self._buffer.write(text)
        self._length += len(text)
        return self

    def append_line(self, text: str = "") -> EfficientStringBuilder:
        """Append text followed by newline.

        Args:
            text: Text to append (empty for blank line)

        Returns:
            Self for method chaining
        """
        self._buffer.write(text)
        self._buffer.write("\n")
        self._length += len(text) + 1
        return self

    def append_lines(self, lines: Iterable[str]) -> EfficientStringBuilder:
        """Append multiple lines.

        Args:
            lines: Iterable of lines

        Returns:
            Self for method chaining
        """
        for line in lines:
            self.append_line(line)
        return self

    def append_format(self, template: str, *args: Any, **kwargs: Any) -> EfficientStringBuilder:
        """Append formatted text using str.format().

        Args:
            template: Format string
            *args: Positional format arguments
            **kwargs: Keyword format arguments

        Returns:
            Self for method chaining
        """
        text = template.format(*args, **kwargs)
        return self.append(text)

    def prepend(self, text: str) -> EfficientStringBuilder:
        """Prepend text (less efficient, use sparingly).

        Args:
            text: Text to prepend

        Returns:
            Self for method chaining
        """
        current = self._buffer.getvalue()
        self._buffer = StringIO()
        self._buffer.write(text)
        self._buffer.write(current)
        self._length += len(text)
        return self

    def get(self) -> str:
        """Get built string without consuming buffer."""
        return self._buffer.getvalue()

    def build(self) -> str:
        """Alias for get() - get the final string."""
        return self._buffer.getvalue()

    def clear(self) -> EfficientStringBuilder:
        """Clear all content.

        Returns:
            Self for method chaining
        """
        self._buffer = StringIO()
        self._length = 0
        return self

    def __str__(self) -> str:
        """Convert to string."""
        return self.get()

    def __len__(self) -> int:
        """Return current length."""
        return self._length

    def __repr__(self) -> str:
        """Return representation."""
        content = self.get()
        if len(content) > 50:
            return f"EfficientStringBuilder({content[:50]!r}...)"
        return f"EfficientStringBuilder({content!r})"


# =============================================================================
# Table and Report Building
# =============================================================================


def build_table(
    headers: List[str],
    rows: List[List[Any]],
    column_widths: Optional[List[int]] = None,
    align: str = "left",
    border: bool = True,
) -> str:
    """Build formatted table efficiently using StringIO.

    Args:
        headers: Column headers
        rows: Table rows (list of cells per row)
        column_widths: Optional custom column widths (auto-calculated if None)
        align: Text alignment ('left', 'right', 'center')
        border: Whether to include table borders

    Returns:
        Formatted table string
    """
    if not headers:
        return ""

    # Calculate column widths if not provided
    if column_widths is None:
        column_widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(column_widths):
                    column_widths[i] = max(column_widths[i], len(str(cell)))

    # Alignment function
    def align_cell(text: str, width: int) -> str:
        if align == "right":
            return text.rjust(width)
        elif align == "center":
            return text.center(width)
        return text.ljust(width)

    builder = EfficientStringBuilder()

    # Create separator for bordered tables
    separator = "+" + "+".join("-" * (w + 2) for w in column_widths) + "+"

    if border:
        builder.append_line(separator)

    # Build header row
    header_cells = [f" {align_cell(str(h), column_widths[i])} " for i, h in enumerate(headers)]
    if border:
        builder.append_line("|" + "|".join(header_cells) + "|")
        builder.append_line(separator)
    else:
        builder.append_line(" | ".join(str(h).ljust(w) for h, w in zip(headers, column_widths)))
        builder.append_line("-" * sum(column_widths))

    # Build data rows
    for row in rows:
        row_cells = [
            f" {align_cell(str(cell), column_widths[i])} "
            for i, cell in enumerate(row)
            if i < len(column_widths)
        ]
        if border:
            builder.append_line("|" + "|".join(row_cells) + "|")
        else:
            builder.append_line(
                " | ".join(str(cell).ljust(w) for cell, w in zip(row, column_widths))
            )

    if border:
        builder.append_line(separator)

    return builder.get()


def build_report(
    title: str,
    sections: Dict[str, Any],
    width: int = 70,
    include_summary: bool = True,
) -> str:
    """Build a formatted report.

    Args:
        title: Report title
        sections: Dict of section_name -> section_content
        width: Width of the report
        include_summary: Whether to include summary section

    Returns:
        Formatted report string
    """
    builder = EfficientStringBuilder()

    # Title
    builder.append_line("=" * width)
    builder.append_line(title.center(width))
    builder.append_line("=" * width)
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
        builder.append_line("=" * width)
        builder.append_line(f"Total sections: {len(sections)}")
        builder.append_line("=" * width)

    return builder.get()


def join_lines(
    items: Iterable[str],
    separator: str = "\n",
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Join items with separator and optional prefix/suffix.

    Uses separator.join() for optimal performance.

    Args:
        items: Items to join
        separator: Separator between items
        prefix: Prefix for entire result
        suffix: Suffix for entire result

    Returns:
        Joined string
    """
    content = separator.join(str(item) for item in items)
    if prefix or suffix:
        return f"{prefix}{content}{suffix}"
    return content


# =============================================================================
# Metrics and Size Formatting
# =============================================================================


def format_metrics(metrics: Dict[str, Any], precision: int = 4) -> str:
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


def format_size(size_bytes: int) -> str:
    """Format byte size into human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size string (e.g., "1.5 GB")
    """
    if size_bytes == 0:
        return "0 B"

    units = ("B", "KB", "MB", "GB", "TB", "PB")
    unit_index = 0
    size = float(size_bytes)

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    return f"{size:.1f} {units[unit_index]}"


# =============================================================================
# Text Processing with Caching
# =============================================================================


@lru_cache(maxsize=1024)
def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = False,
    normalize_unicode: bool = True,
    normalize_spaces: bool = True,
) -> str:
    """Normalize text with caching for repeated operations.

    Args:
        text: Input text to normalize
        lowercase: Convert to lowercase
        remove_punctuation: Remove punctuation marks
        normalize_unicode: Normalize unicode characters
        normalize_spaces: Normalize whitespace

    Returns:
        Normalized text
    """
    if not text:
        return ""

    result = text

    # Normalize unicode characters first
    if normalize_unicode:
        result = unicodedata.normalize("NFKC", result)
        result = result.translate(_NORMALIZE_TRANSLATOR)
        result = _COMPILED_PATTERNS["unicode_quotes"].sub('"', result)
        result = _COMPILED_PATTERNS["unicode_spaces"].sub(" ", result)

    # Remove HTML tags
    result = _COMPILED_PATTERNS["html_tags"].sub(" ", result)

    # Normalize spaces
    if normalize_spaces:
        result = _COMPILED_PATTERNS["whitespace"].sub(" ", result)
        result = result.strip()

    # Remove punctuation
    if remove_punctuation:
        result = result.translate(_PUNCTUATION_TRANSLATOR)

    # Convert to lowercase
    if lowercase:
        result = result.lower()

    return result


def clean_text(
    text: str,
    remove_html: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    normalize_spaces: bool = True,
) -> str:
    """Clean text by removing unwanted elements.

    Args:
        text: Input text
        remove_html: Remove HTML tags
        remove_urls: Remove URLs
        remove_emails: Remove email addresses
        normalize_spaces: Normalize whitespace

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    result = text

    if remove_html:
        result = _COMPILED_PATTERNS["html_tags"].sub(" ", result)

    if remove_urls:
        result = _COMPILED_PATTERNS["url"].sub(" ", result)

    if remove_emails:
        result = _COMPILED_PATTERNS["email"].sub(" ", result)

    if normalize_spaces:
        result = _COMPILED_PATTERNS["whitespace"].sub(" ", result)
        result = result.strip()

    return result


@lru_cache(maxsize=512)
def extract_tokens(
    text: str,
    min_length: int = 1,
    max_length: int = 50,
    normalize: bool = True,
) -> tuple[str, ...]:
    """Extract and cache tokens from text.

    Note: Returns tuple for hashability (required for caching).

    Args:
        text: Input text
        min_length: Minimum token length
        max_length: Maximum token length
        normalize: Whether to normalize text first

    Returns:
        Tuple of tokens
    """
    if normalize:
        text = normalize_text(text, lowercase=True, remove_punctuation=True)

    # Split on whitespace and filter by length
    tokens = tuple(token for token in text.split() if min_length <= len(token) <= max_length)

    return tokens


def batch_normalize_text(texts: Iterable[str], **kwargs: Any) -> List[str]:
    """Efficiently normalize multiple texts using batch processing.

    Args:
        texts: Iterable of texts to normalize
        **kwargs: Arguments passed to normalize_text

    Returns:
        List of normalized texts
    """
    return [normalize_text(text, **kwargs) for text in texts]


# Legacy aliases for backward compatibility
def tokenize(text: str) -> List[str]:
    """Tokenize text into words (legacy alias for extract_tokens)."""
    return list(extract_tokens(text, normalize=True))


def strip_punctuation(text: str) -> str:
    """Remove punctuation from text (legacy function)."""
    return text.translate(_PUNCTUATION_TRANSLATOR)


# =============================================================================
# String Utilities
# =============================================================================


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
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

    if len(suffix) >= max_length:
        return suffix[:max_length]

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
    return "\n".join(indent_str + line for line in text.split("\n"))


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
    if not words:
        return ""

    lines: List[str] = []
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


# =============================================================================
# JSON Utilities
# =============================================================================


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
        return json.dumps(data, indent=indent, default=str, ensure_ascii=False)
    return json.dumps(data, default=str, ensure_ascii=False)


def parse_jsonl(text: str) -> List[Dict[str, Any]]:
    """Parse JSONL (JSON Lines) format.

    Args:
        text: JSONL text

    Returns:
        List of parsed dictionaries
    """
    results: List[Dict[str, Any]] = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line:
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
        builder.append_line(json.dumps(item, default=str, ensure_ascii=False))
    return builder.get()


# =============================================================================
# Miscellaneous Utilities
# =============================================================================


def sanitize_filename(filename: str) -> str:
    """Sanitize string for use as filename.

    Args:
        filename: Filename to sanitize

    Returns:
        Sanitized filename
    """
    # Replace invalid characters
    sanitized = _COMPILED_PATTERNS["invalid_filename"].sub("_", filename)
    # Replace multiple underscores with single
    sanitized = _COMPILED_PATTERNS["multiple_underscores"].sub("_", sanitized)
    # Remove leading/trailing underscores and spaces
    return sanitized.strip("_ ")


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
        Comparison dictionary with equality, lengths, and first difference
    """
    first_diff = next((i for i, (c1, c2) in enumerate(zip(str1, str2)) if c1 != c2), -1)

    return {
        "equal": str1 == str2,
        "len_str1": len(str1),
        "len_str2": len(str2),
        "first_diff_pos": first_diff,
        "preview_str1": str1[:100] if len(str1) > 100 else str1,
        "preview_str2": str2[:100] if len(str2) > 100 else str2,
    }
