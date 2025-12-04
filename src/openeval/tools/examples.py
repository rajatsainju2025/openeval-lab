"""Example tool implementations and utilities for the Tool framework."""

from __future__ import annotations

import time
import json
from typing import Any
from pathlib import Path

from .base import Tool, ToolResult


class FileReaderTool(Tool):
    """Tool for reading file contents."""

    name = "file_reader"
    description = "Read contents of a text file from the filesystem."

    def __init__(self, max_size_bytes: int = 10_000_000):
        """Initialize with maximum file size limit."""
        self.max_size_bytes = max_size_bytes

    def run(self, query: str, **kwargs: Any) -> ToolResult:
        """Read file specified in query.

        Args:
            query: Path to the file to read
            **kwargs: Optional encoding parameter

        Returns:
            ToolResult with file contents or error
        """
        start = time.perf_counter()
        encoding = kwargs.get("encoding", "utf-8")

        try:
            path = Path(query).expanduser().resolve()

            if not path.exists():
                return ToolResult(
                    False,
                    None,
                    f"File not found: {path}",
                    (time.perf_counter() - start) * 1000,
                )

            if not path.is_file():
                return ToolResult(
                    False,
                    None,
                    f"Path is not a file: {path}",
                    (time.perf_counter() - start) * 1000,
                )

            size = path.stat().st_size
            if size > self.max_size_bytes:
                return ToolResult(
                    False,
                    None,
                    f"File too large: {size} bytes (max: {self.max_size_bytes})",
                    (time.perf_counter() - start) * 1000,
                )

            content = path.read_text(encoding=encoding)
            return ToolResult(True, content, None, (time.perf_counter() - start) * 1000)

        except UnicodeDecodeError as e:
            return ToolResult(
                False,
                None,
                f"Failed to decode file: {e}",
                (time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                False, None, f"Error reading file: {e}", (time.perf_counter() - start) * 1000
            )


class JSONParserTool(Tool):
    """Tool for parsing JSON strings."""

    name = "json_parser"
    description = "Parse JSON string and return structured data."

    def run(self, query: str, **kwargs: Any) -> ToolResult:
        """Parse JSON from query string.

        Args:
            query: JSON string to parse
            **kwargs: Optional strict parameter

        Returns:
            ToolResult with parsed JSON or error
        """
        start = time.perf_counter()
        strict = kwargs.get("strict", True)

        try:
            parsed = json.loads(query, strict=strict)
            return ToolResult(True, parsed, None, (time.perf_counter() - start) * 1000)
        except json.JSONDecodeError as e:
            return ToolResult(
                False,
                None,
                f"Invalid JSON: {e.msg} at line {e.lineno}, column {e.colno}",
                (time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                False, None, f"Error parsing JSON: {e}", (time.perf_counter() - start) * 1000
            )


class StringTransformTool(Tool):
    """Tool for common string transformations."""

    name = "string_transform"
    description = "Apply common string transformations (upper, lower, title, reverse, etc.)."

    TRANSFORMS = {
        "upper": str.upper,
        "lower": str.lower,
        "title": str.title,
        "capitalize": str.capitalize,
        "reverse": lambda s: s[::-1],
        "strip": str.strip,
        "length": len,
    }

    def run(self, query: str, **kwargs: Any) -> ToolResult:
        """Transform string according to specified operation.

        Args:
            query: String to transform
            **kwargs: transform parameter specifying operation

        Returns:
            ToolResult with transformed string or error
        """
        start = time.perf_counter()
        transform = kwargs.get("transform", "lower")

        if transform not in self.TRANSFORMS:
            return ToolResult(
                False,
                None,
                f"Unknown transform: {transform}. Available: {list(self.TRANSFORMS.keys())}",
                (time.perf_counter() - start) * 1000,
            )

        try:
            result = self.TRANSFORMS[transform](query)
            return ToolResult(True, result, None, (time.perf_counter() - start) * 1000)
        except Exception as e:
            return ToolResult(
                False,
                None,
                f"Error applying transform: {e}",
                (time.perf_counter() - start) * 1000,
            )


__all__ = ["FileReaderTool", "JSONParserTool", "StringTransformTool"]
