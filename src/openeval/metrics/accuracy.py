from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


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

        # Use NumPy for vectorized comparison if available
        if HAS_NUMPY and np is not None:
            pred_array = np.array([str(p).strip() for p in preds])
            ref_array = np.array([str(r).strip() for r in refs])
            correct = np.sum(pred_array == ref_array)
        else:
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

        def f1(pred_tokens: list[str], ref_tokens: list[str]) -> float:
            """Calculate F1 for a single prediction-reference pair."""
            if not pred_tokens and not ref_tokens:
                return 1.0
            if not pred_tokens or not ref_tokens:
                return 0.0

            # multiset overlap (bag of words) - use pre-computed Counters
            cp, cr = Counter(pred_tokens), Counter(ref_tokens)
            overlap = sum((cp & cr).values())
            prec = overlap / sum(cp.values())
            rec = overlap / sum(cr.values())

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

        # Pre-compute all token lists in one pass to avoid repeated normalization
        pred_tokens_list = [str(p).strip().split() for p in preds]
        ref_tokens_list = [str(r).strip().split() for r in refs]

        # Compute F1 scores with vectorized computation if NumPy available
        if HAS_NUMPY and np is not None and len(preds) > 100:
            # For large datasets, use NumPy-accelerated computation
            scores = []
            for pt, rt in zip(pred_tokens_list, ref_tokens_list):
                scores.append(f1(pt, rt))
            score = float(np.mean(scores))
        else:
            # For small datasets, use pure Python (no NumPy overhead)
            score = sum(f1(pt, rt) for pt, rt in zip(pred_tokens_list, ref_tokens_list)) / len(
                preds
            )

        return {"f1": score}
