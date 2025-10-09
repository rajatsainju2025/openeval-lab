"""Metric protocol for evaluation scoring."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol


class Metric(Protocol):
    """Protocol defining evaluation metrics for model outputs.

    A Metric computes quantitative scores comparing model predictions against reference
    answers. Metrics can range from simple exact match to complex semantic similarity
    measures. They must be deterministic and return consistent scores for the same
    inputs.

    Common metric types include:
    - Accuracy metrics (exact match, case-insensitive match)
    - Partial match metrics (F1 score, ROUGE)
    - Semantic metrics (BERTScore, embedding similarity)
    - Task-specific metrics (BLEU for translation, perplexity for LMs)

    Invariants:
        - Must be deterministic given same inputs
        - Must handle batched inputs efficiently
        - Should be robust to common input variations
        - Should validate inputs and raise informative errors

    Attributes:
        name: A unique identifier for this metric.
    """

    name: str

    def compute(self, predictions: Iterable[Any], references: Iterable[Any]) -> Mapping[str, float]:
        """Compute evaluation scores comparing predictions to references.

        Both inputs must be iterables of the same length. The metric may compute
        multiple related scores (e.g., precision/recall/F1) and return them in
        a dictionary.

        Args:
            predictions: Model outputs to evaluate.
            references: Expected correct outputs to compare against.

        Returns:
            Dictionary mapping score names to float values.
            Common keys include:
            - accuracy: Fraction of exact matches
            - f1: F1 score for partial matches
            - rouge_1/2/L: ROUGE scores for summarization
            - bleu: BLEU score for translation

        Raises:
            ValueError: If inputs are invalid or incompatible.
        """
        ...


__all__ = ["Metric"]
