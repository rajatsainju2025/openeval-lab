from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..core import Dataset, Example


@dataclass
class JSONLinesDataset(Dataset):
    path: Path
    name: str = "jsonl"
    text_field: str = "input"
    ref_field: str = "reference"

    def __iter__(self) -> Iterator[Example]:
        with open(self.path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                yield Example(id=str(obj.get("id", i)), input=obj[self.text_field], reference=obj[self.ref_field], meta=obj)
