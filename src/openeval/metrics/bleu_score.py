"""
BLEU Score Metric for OpenEval Lab

This module implements BLEU (Bilingual Evaluation Understudy) score calculation
for evaluating machine translation and text generation quality.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, List, Mapping, Optional, Union

import logging

logger = logging.getLogger(__name__)


class BLEUScore:
    """
    BLEU Score implementation for text generation evaluation.

    BLEU measures the similarity between machine-generated text and reference text
    by comparing n-grams of various lengths.
    """

    def __init__(
        self,
        n_gram: int = 4,
        weights: Optional[List[float]] = None,
        smoothing: bool = True,
        case_sensitive: bool = False,
    ):
        """
        Initialize BLEU scorer.

        Args:
            n_gram: Maximum n-gram order to use (default: 4)
            weights: Weights for different n-gram orders. If None, uniform weights are used.
            smoothing: Whether to apply smoothing to avoid zero counts
            case_sensitive: Whether to preserve case in text processing
        """
        self.n_gram = n_gram
        self.weights = weights or [1.0 / n_gram] * n_gram
        self.smoothing = smoothing
        self.case_sensitive = case_sensitive

        if len(self.weights) != n_gram:
            raise ValueError(
                f"Number of weights ({len(self.weights)}) must match n_gram ({n_gram})"
            )

        if abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

    def compute(
        self, predictions: Iterable[str], references: Iterable[Union[str, List[str]]]
    ) -> Mapping[str, float]:
        """
        Compute BLEU score for predictions against references.

        Args:
            predictions: List of predicted texts
            references: List of reference texts (can be single string or list of strings per prediction)

        Returns:
            Dictionary with BLEU scores and component metrics
        """
        predictions = list(predictions)
        references = list(references)

        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must be equal")

        if len(predictions) == 0:
            return {
                "bleu": 0.0,
                "bleu_1gram": 0.0,
                "bleu_2gram": 0.0,
                "bleu_3gram": 0.0,
                "bleu_4gram": 0.0,
                "brevity_penalty": 0.0,
                "precision_1gram": 0.0,
                "precision_2gram": 0.0,
                "precision_3gram": 0.0,
                "precision_4gram": 0.0,
            }

        # Process each prediction-reference pair
        total_bleu = 0.0
        total_bp = 0.0
        precision_scores = [0.0] * self.n_gram

        for pred, ref in zip(predictions, references):
            # Handle multiple references
            if isinstance(ref, str):
                ref_texts = [ref]
            else:
                ref_texts = ref

            # Preprocess texts
            pred_tokens = self._preprocess_text(pred)
            ref_tokens_list = [self._preprocess_text(r) for r in ref_texts]

            # Calculate BLEU for this pair
            bleu_score, bp, precisions = self._compute_bleu_single(pred_tokens, ref_tokens_list)
            total_bleu += bleu_score
            total_bp += bp

            for i, prec in enumerate(precisions):
                precision_scores[i] += prec

        # Average scores
        num_pairs = len(predictions)
        avg_bleu = total_bleu / num_pairs
        avg_bp = total_bp / num_pairs
        avg_precisions = [p / num_pairs for p in precision_scores]

        return {
            "bleu": avg_bleu,
            "bleu_1gram": avg_precisions[0] if len(avg_precisions) > 0 else 0.0,
            "bleu_2gram": avg_precisions[1] if len(avg_precisions) > 1 else 0.0,
            "bleu_3gram": avg_precisions[2] if len(avg_precisions) > 2 else 0.0,
            "bleu_4gram": avg_precisions[3] if len(avg_precisions) > 3 else 0.0,
            "brevity_penalty": avg_bp,
            "precision_1gram": avg_precisions[0] if len(avg_precisions) > 0 else 0.0,
            "precision_2gram": avg_precisions[1] if len(avg_precisions) > 1 else 0.0,
            "precision_3gram": avg_precisions[2] if len(avg_precisions) > 2 else 0.0,
            "precision_4gram": avg_precisions[3] if len(avg_precisions) > 3 else 0.0,
        }

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text by tokenizing and optionally lowercasing."""
        if not self.case_sensitive:
            text = text.lower()
        # Simple tokenization - split on whitespace and punctuation
        import re

        tokens = re.findall(r"\w+", text)
        return tokens

    def _compute_bleu_single(
        self, pred_tokens: List[str], ref_tokens_list: List[List[str]]
    ) -> tuple[float, float, List[float]]:
        """
        Compute BLEU score for a single prediction-reference pair.

        Returns:
            Tuple of (bleu_score, brevity_penalty, precision_scores)
        """
        if not pred_tokens:
            return 0.0, 0.0, [0.0] * self.n_gram

        # Find the reference with minimum length difference
        ref_lengths = [len(ref) for ref in ref_tokens_list]
        pred_len = len(pred_tokens)
        closest_ref_len = min(ref_lengths, key=lambda x: abs(x - pred_len))
        closest_ref_idx = ref_lengths.index(closest_ref_len)
        closest_ref = ref_tokens_list[closest_ref_idx]

        # Calculate brevity penalty
        bp = self._brevity_penalty(pred_len, closest_ref_len)

        # Calculate precision for each n-gram order
        precisions = []
        for n in range(1, self.n_gram + 1):
            precision = self._ngram_precision(pred_tokens, ref_tokens_list, n)
            precisions.append(precision)

        # Apply smoothing if enabled
        if self.smoothing:
            precisions = self._apply_smoothing(precisions)

        # Calculate BLEU score
        if any(p == 0 for p in precisions):
            bleu = 0.0
        else:
            log_precision_sum = sum(w * math.log(p) for w, p in zip(self.weights, precisions))
            bleu = bp * math.exp(log_precision_sum)

        return bleu, bp, precisions

    def _ngram_precision(
        self, pred_tokens: List[str], ref_tokens_list: List[List[str]], n: int
    ) -> float:
        """Calculate n-gram precision."""
        if len(pred_tokens) < n:
            return 0.0

        # Count n-grams in prediction
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        pred_counts = Counter(pred_ngrams)

        # Count maximum occurrences in any reference
        max_ref_counts = Counter()
        for ref_tokens in ref_tokens_list:
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            ref_counts = Counter(ref_ngrams)
            for ngram in ref_counts:
                max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_counts[ngram])

        # Calculate clipped counts
        clipped_counts = 0
        total_pred = 0

        for ngram, count in pred_counts.items():
            clipped_count = min(count, max_ref_counts[ngram])
            clipped_counts += clipped_count
            total_pred += count

        if total_pred == 0:
            return 0.0

        return clipped_counts / total_pred

    def _get_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        """Extract n-grams from token list."""
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def _brevity_penalty(self, pred_len: int, ref_len: int) -> float:
        """Calculate brevity penalty."""
        if pred_len > ref_len:
            return 1.0
        elif pred_len == 0:
            return 0.0
        else:
            return math.exp(1 - ref_len / pred_len)

    def _apply_smoothing(self, precisions: List[float]) -> List[float]:
        """Apply smoothing to precision scores to avoid zero values."""
        smoothed = []
        for i, p in enumerate(precisions):
            if p == 0.0:
                # Use a small epsilon or average of other precisions
                non_zero = [p for p in precisions if p > 0]
                if non_zero:
                    smoothed_p = sum(non_zero) / len(non_zero) * 0.1  # 10% of average
                else:
                    smoothed_p = 1e-7  # Very small epsilon
            else:
                smoothed_p = p
            smoothed.append(smoothed_p)
        return smoothed


def compute_bleu_score(
    predictions: Iterable[str], references: Iterable[Union[str, List[str]]], **kwargs: Any
) -> Mapping[str, float]:
    """
    Convenience function to compute BLEU score.

    Args:
        predictions: Predicted texts
        references: Reference texts
        **kwargs: Additional arguments passed to BLEUScore

    Returns:
        BLEU score metrics
    """
    bleu = BLEUScore(**kwargs)
    return bleu.compute(predictions, references)
