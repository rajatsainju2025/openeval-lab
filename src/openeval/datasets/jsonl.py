from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..core import Dataset, Example


@dataclass
class JSONLinesDataset(Dataset):
    """Dataset loader for JSONL (JSON Lines) format files.

    Each line in the file should be a valid JSON object containing at minimum
    the input and reference fields. Additional fields are stored in metadata.

    Args:
        path: Path to the JSONL file
        name: Name identifier for the dataset
        text_field: Name of the field containing the input text (default: "input")
        ref_field: Name of the field containing the reference answer (default: "reference")

    Raises:
        FileNotFoundError: If the specified file does not exist
        json.JSONDecodeError: If a line contains invalid JSON
        KeyError: If required fields are missing from an example
    """

    path: Path
    name: str = "jsonl"
    text_field: str = "input"
    ref_field: str = "reference"

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

    def __iter__(self) -> Iterator[Example]:
        """Iterate through examples in the dataset.

        Yields:
            Example objects with id, input, reference, and metadata

        Raises:
            json.JSONDecodeError: If a line contains invalid JSON
            KeyError: If required fields are missing
        """
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise json.JSONDecodeError(
                            f"Invalid JSON on line {i} of {self.path}: {e.msg}",
                            e.doc,
                            e.pos,
                        ) from e

                    # Validate required fields
                    if self.text_field not in obj:
                        raise KeyError(
                            f"Missing required field '{self.text_field}' on line {i} of {self.path}\n"
                            f"Available fields: {list(obj.keys())}"
                        )
                    if self.ref_field not in obj:
                        raise KeyError(
                            f"Missing required field '{self.ref_field}' on line {i} of {self.path}\n"
                            f"Available fields: {list(obj.keys())}"
                        )

                    yield Example(
                        id=str(obj.get("id", i)),
                        input=obj[self.text_field],
                        reference=obj[self.ref_field],
                        meta=obj,
                    )
        except (OSError, IOError) as e:
            raise IOError(
                f"Error reading dataset file {self.path}: {e}\n"
                f"Please check file permissions and disk space."
            ) from e
