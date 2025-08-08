from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass
class ExactMatch:
    name: str = "exact_match"

    def compute(self, predictions: Iterable[str], references: Iterable[str]) -> Mapping[str, float]:
        preds = list(predictions)
        refs = list(references)
        correct = sum(1 for p, r in zip(preds, refs) if str(p).strip() == str(r).strip())
        total = len(preds) if preds else 1
        return {"accuracy": correct / total}
