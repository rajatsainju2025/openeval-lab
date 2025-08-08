from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from ..core import Dataset, Example

try:  # optional dependency
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore


@dataclass
class HFDataset(Dataset):
    dataset_name: str
    split: str = "train"
    subset: Optional[str] = None
    input_field: str = "question"
    reference_field: str = "answers"
    name: str = "hf"

    def __iter__(self) -> Iterator[Example]:  # pragma: no cover - depends on HF datasets
        if load_dataset is None:
            raise RuntimeError("Please install openeval-lab[hf] to use HFDataset.")
        ds = load_dataset(self.dataset_name, self.subset, split=self.split)
        for i, row in enumerate(ds):
            ref = row[self.reference_field]
            # handle common formats e.g., SQuAD answers: {text: [..]}
            if isinstance(ref, dict) and "text" in ref and isinstance(ref["text"], list):
                reference = ref["text"][0] if ref["text"] else ""
            else:
                reference = ref
            yield Example(
                id=str(row.get("id", i)), input=row[self.input_field], reference=reference, meta=row
            )
