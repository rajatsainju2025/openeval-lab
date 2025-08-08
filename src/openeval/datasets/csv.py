from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..core import Dataset, Example


@dataclass
class CSVDataset(Dataset):
    path: Path
    name: str = "csv"
    text_field: str = "input"
    ref_field: str = "reference"

    def __iter__(self) -> Iterator[Example]:
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                yield Example(
                    id=str(row.get("id", i)),
                    input=row[self.text_field],
                    reference=row[self.ref_field],
                    meta=row,
                )
