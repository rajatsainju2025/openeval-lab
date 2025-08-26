from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Mapping, Any, Optional, Dict

from ..core import Dataset, Example


@dataclass
class InlineDataset(Dataset):
    name: str = "inline"
    examples: Optional[List[Dict[str, Any]]] = None

    def __iter__(self) -> Iterator[Example]:
        examples = self.examples or []
        for i, obj in enumerate(examples):
            yield Example(
                id=str(obj.get("id", i)),
                input=obj.get("input"),
                reference=obj.get("reference"),
                meta=dict(obj),
            )

    def __len__(self) -> int:
        return len(self.examples or [])
