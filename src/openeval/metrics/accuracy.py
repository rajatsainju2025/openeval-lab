from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass
class ExactMatch:
    """Compute exact match accuracy between predictions and references.

    Measures the percentage of predictions that exactly match their corresponding
    references after stripping whitespace. Case-sensitive by default.

    Args:
        name: Metric identifier (default: "exact_match")

    Returns:
        Dictionary with "accuracy" key containing the exact match rate (0.0 to 1.0)
    """

    name: str = "exact_match"

    def compute(self, predictions: Iterable[str], references: Iterable[str]) -> Mapping[str, float]:
        """Compute exact match accuracy.

        Args:
            predictions: Iterable of predicted strings
            references: Iterable of reference strings

        Returns:
            Dictionary with "accuracy" score

        Raises:
            ValueError: If predictions and references have different lengths
        """
        preds = list(predictions)
        refs = list(references)

        if len(preds) != len(refs):
            raise ValueError(
                f"Predictions and references must have the same length. "
                f"Got {len(preds)} predictions and {len(refs)} references."
            )

        if not preds:
            return {"accuracy": 0.0}

        correct = sum(1 for p, r in zip(preds, refs) if str(p).strip() == str(r).strip())
        total = len(preds)
        return {"accuracy": correct / total}


@dataclass
class TokenF1:
    """Compute token-level F1 score between predictions and references.

    Treats predictions and references as bags of words and computes F1 score
    based on token overlap. Useful for tasks where word order is less important.

    Args:
        name: Metric identifier (default: "token_f1")

    Returns:
        Dictionary with "f1" key containing the F1 score (0.0 to 1.0)
    """

    name: str = "token_f1"

    def compute(self, predictions: Iterable[str], references: Iterable[str]) -> Mapping[str, float]:
        """Compute token-level F1 score.

        Args:
            predictions: Iterable of predicted strings
            references: Iterable of reference strings

        Returns:
            Dictionary with "f1" score

        Raises:
            ValueError: If predictions and references have different lengths
        """

        def f1(p: str, r: str) -> float:
            """Calculate F1 for a single prediction-reference pair."""
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

        if len(preds) != len(refs):
            raise ValueError(
                f"Predictions and references must have the same length. "
                f"Got {len(preds)} predictions and {len(refs)} references."
            )

        if not preds:
            return {"f1": 0.0}

        score = sum(f1(p, r) for p, r in zip(preds, refs)) / len(preds)
        return {"f1": score}
