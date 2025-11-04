"""Shared metric normalization utilities.

This module provides reusable normalization helpers to avoid repeated
text processing across multiple metric computations.
"""

from typing import Dict, List, Tuple
from collections import Counter


class NormalizationCache:
    """Cache for normalized text to avoid recomputation."""

    def __init__(self):
        self._cache: Dict[str, List[str]] = {}

    def normalize_and_tokenize(self, text: str) -> List[str]:
        """Normalize text (strip/lowercase) and tokenize.

        Args:
            text: Input text to normalize

        Returns:
            List of normalized tokens
        """
        if text in self._cache:
            return self._cache[text]

        tokens = str(text).strip().lower().split()
        self._cache[text] = tokens
        return tokens

    def clear(self):
        """Clear the normalization cache."""
        self._cache.clear()


def normalize_batch_predictions(
    predictions: List[str],
    lowercase: bool = False,
) -> Tuple[List[str], List[List[str]]]:
    """Normalize predictions and compute token lists in one pass.

    Args:
        predictions: List of prediction strings
        lowercase: Whether to convert to lowercase

    Returns:
        Tuple of (normalized_strings, token_lists)
    """
    normalized = []
    tokens_list = []

    for pred in predictions:
        text = str(pred).strip()
        if lowercase:
            text = text.lower()
        normalized.append(text)
        tokens_list.append(text.split())

    return normalized, tokens_list


def compute_token_overlap(pred_tokens: List[str], ref_tokens: List[str]) -> Tuple[int, int, int]:
    """Compute token overlap metrics for bag-of-words comparison.

    Args:
        pred_tokens: Prediction tokens
        ref_tokens: Reference tokens

    Returns:
        Tuple of (overlap_count, pred_total, ref_total)
    """
    if not pred_tokens or not ref_tokens:
        return 0, len(pred_tokens), len(ref_tokens)

    cp = Counter(pred_tokens)
    cr = Counter(ref_tokens)
    overlap = sum((cp & cr).values())

    return overlap, sum(cp.values()), sum(cr.values())
