from __future__ import annotations

from typing import Iterable, Mapping


class ROUGEL:
    name = "rouge_l"

    def __init__(self) -> None:
        try:
            from rouge_score import rouge_scorer  # noqa: F401
        except Exception as e:  # pragma: no cover - optional dep
            raise RuntimeError("rouge-score not installed; install with `pip install -e '.[metrics]'`") from e

    def compute(self, predictions: Iterable[str], references: Iterable[str]) -> Mapping[str, float]:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(ref or "", pred or "") for pred, ref in zip(predictions, references)]
        # average F-measure
        vals = [s["rougeL"].fmeasure for s in scores]
        if not vals:
            return {"rougeL_f": 0.0}
        return {"rougeL_f": float(sum(vals) / len(vals))}
