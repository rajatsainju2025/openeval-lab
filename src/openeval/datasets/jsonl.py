from __future__ import annotations

import json
import mmap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

from ..core import Dataset, Example


@dataclass
class JSONLinesDataset(Dataset):
    """Dataset loader for JSONL (JSON Lines) format files with streaming support.

    Each line in the file should be a valid JSON object containing at minimum
    the input and reference fields. Additional fields are stored in metadata.

    This implementation uses memory-efficient streaming to handle large datasets
    without loading the entire file into memory.

    Args:
        path: Path to the JSONL file
        name: Name identifier for the dataset
        text_field: Name of the field containing the input text (default: "input")
        ref_field: Name of the field containing the reference answer (default: "reference")
        chunk_size: Number of bytes to read at once for chunked streaming (default: 65536)
        use_mmap: Whether to use memory-mapped file access for large files (default: False)
        mmap_threshold_mb: File size threshold in MB to trigger mmap (default: 100)

    Raises:
        FileNotFoundError: If the specified file does not exist
        json.JSONDecodeError: If a line contains invalid JSON
        KeyError: If required fields are missing from an example
    """

    path: Path
    name: str = "jsonl"
    text_field: str = "input"
    ref_field: str = "reference"
    chunk_size: int = 65536  # 64KB chunks for efficient I/O
    use_mmap: bool = False
    mmap_threshold_mb: int = 100

    def __post_init__(self):
        """Validate dataset path and configuration."""
        if not isinstance(self.path, Path):
            self.path = Path(self.path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.path}\n"
                f"Please ensure the file exists and the path is correct."
            )

        if not self.path.is_file():
            raise ValueError(
                f"Path is not a file: {self.path}\n" f"Expected a JSONL file, but got a directory."
            )

        # Auto-enable mmap for large files
        if not self.use_mmap:
            file_size_mb = self.path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.mmap_threshold_mb:
                self.use_mmap = True

    def __iter__(self) -> Iterator[Example]:
        """Iterate through examples in the dataset with streaming.

        Yields:
            Example objects with id, input, reference, and metadata

        Raises:
            json.JSONDecodeError: If a line contains invalid JSON
            KeyError: If required fields are missing
        """
        if self.use_mmap:
            yield from self._iter_mmap()
        else:
            yield from self._iter_chunked()

    def _iter_chunked(self) -> Iterator[Example]:
        """Iterate using chunked reading for memory efficiency."""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                buffer = ""
                line_num = 0

                while True:
                    # Read chunk
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        # Process final buffer
                        if buffer.strip():
                            line_num += 1
                            yield self._parse_line(buffer, line_num)
                        break

                    buffer += chunk
                    lines = buffer.split("\n")

                    # Keep last incomplete line in buffer
                    buffer = lines[-1]
                    lines = lines[:-1]

                    for line in lines:
                        if line.strip():  # Skip empty lines
                            line_num += 1
                            yield self._parse_line(line, line_num)

        except (OSError, IOError) as e:
            raise IOError(
                f"Error reading dataset file {self.path}: {e}\n"
                f"Please check file permissions and disk space."
            ) from e

    def _iter_mmap(self) -> Iterator[Example]:
        """Iterate using memory-mapped file for large files."""
        try:
            with open(self.path, "r+b") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    buffer = b""
                    line_num = 0

                    for chunk in iter(lambda: mm.read(self.chunk_size), b""):
                        buffer += chunk
                        lines = buffer.split(b"\n")

                        # Keep last incomplete line in buffer
                        buffer = lines[-1]
                        lines = lines[:-1]

                        for line_bytes in lines:
                            line = line_bytes.decode("utf-8")
                            if line.strip():
                                line_num += 1
                                yield self._parse_line(line, line_num)

                    # Process final buffer
                    if buffer.strip():
                        line_num += 1
                        line = buffer.decode("utf-8")
                        yield self._parse_line(line, line_num)

        except (OSError, IOError) as e:
            raise IOError(
                f"Error reading dataset file {self.path}: {e}\n"
                f"Please check file permissions and disk space."
            ) from e

    def _parse_line(self, line: str, line_num: int) -> Example:
        """Parse a single JSONL line into an Example.

        Args:
            line: JSON string to parse
            line_num: Line number for error reporting

        Returns:
            Parsed Example object

        Raises:
            json.JSONDecodeError: If line contains invalid JSON
            KeyError: If required fields are missing
        """
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON on line {line_num} of {self.path}: {e.msg}",
                e.doc,
                e.pos,
            ) from e

        # Validate required fields
        if self.text_field not in obj:
            raise KeyError(
                f"Missing required field '{self.text_field}' on line {line_num} of {self.path}\n"
                f"Available fields: {list(obj.keys())}"
            )
        if self.ref_field not in obj:
            raise KeyError(
                f"Missing required field '{self.ref_field}' on line {line_num} of {self.path}\n"
                f"Available fields: {list(obj.keys())}"
            )

        return Example(
            id=str(obj.get("id", line_num)),
            input=obj[self.text_field],
            reference=obj[self.ref_field],
            meta=obj,
        )

    def count_lines(self) -> int:
        """Count total lines in the dataset efficiently.

        Returns:
            Number of non-empty lines in the file
        """
        count = 0
        with open(self.path, "rb") as f:
            buffer = b""
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                buffer += chunk
                count += buffer.count(b"\n")
                # Keep only the last partial line
                last_newline = buffer.rfind(b"\n")
                if last_newline != -1:
                    buffer = buffer[last_newline + 1 :]
        return count

    def peek(self, n: int = 5) -> List[Example]:
        """Peek at the first n examples without consuming the iterator.

        Args:
            n: Number of examples to peek at

        Returns:
            List of the first n examples
        """
        examples = []
        for i, example in enumerate(self):
            if i >= n:
                break
            examples.append(example)
        return examples
