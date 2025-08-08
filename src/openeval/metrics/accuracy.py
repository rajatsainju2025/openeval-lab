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


@dataclass
class TokenF1:
    name: str = "token_f1"

    def compute(self, predictions: Iterable[str], references: Iterable[str]) -> Mapping[str, float]:
        def f1(p: str, r: str) -> float:
            ps = str(p).strip().split()
            rs = str(r).strip().split()
            if not ps and not rs:
                return 1.0
            if not ps or not rs:
                return 0.0
            # multiset overlap (bag of words)
            from collections import Counter

            cp, cr = Counter(ps), Counter(rs)
            overlap = sum((cp & cr).values())
            prec = overlap / max(1, sum(cp.values()))
            rec = overlap / max(1, sum(cr.values()))
            if prec + rec == 0:
                return 0.0
            return 2 * prec * rec / (prec + rec)

        preds = list(predictions)
        refs = list(references)
        if not preds:
            return {"f1": 0.0}
        score = sum(f1(p, r) for p, r in zip(preds, refs)) / len(preds)
        return {"f1": score}
