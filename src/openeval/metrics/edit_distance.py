from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, List

from ..core import Metric


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    # DP with two rows
    prev = list(range(len(b) + 1))
    cur = [0] * (len(b) + 1)
    for i, ca in enumerate(a, start=1):
        cur[0] = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(
                prev[j] + 1,      # deletion
                cur[j - 1] + 1,    # insertion
                prev[j - 1] + cost # substitution
            )
        prev, cur = cur, prev
    return prev[-1]


@dataclass
class CharEditDistance(Metric):
    name: str = "char_edit_distance"

    def compute(self, predictions: Iterable[Any], references: Iterable[Any]) -> Mapping[str, float]:
        preds: List[str] = [str(p) if p is not None else "" for p in predictions]
        refs: List[str] = [str(r) if r is not None else "" for r in references]
        n = min(len(preds), len(refs))
        if n == 0:
            return {"avg_distance": 0.0, "avg_similarity": 0.0}
        dists = []
        sims = []
        for p, r in zip(preds, refs):
            d = _levenshtein(p, r)
            maxlen = max(len(p), len(r), 1)
            dists.append(d)
            sims.append(1.0 - (d / maxlen))
        return {
            "avg_distance": float(sum(dists) / n),
            "avg_similarity": float(sum(sims) / n),
        }
