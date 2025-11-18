"""Metrics caching with memoization for normalization.

Adds @lru_cache to expensive text normalization operations.
"""

from functools import lru_cache
import re


@lru_cache(maxsize=10000)
def normalize_text_cached(text: str) -> str:
    """Cache text normalization results."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


@lru_cache(maxsize=5000)
def tokenize_cached(text: str) -> tuple:
    """Cache tokenization results."""
    return tuple(text.split())


@lru_cache(maxsize=5000)
def strip_punctuation_cached(text: str) -> str:
    """Cache punctuation stripping."""
    return re.sub(r"[^\w\s]", "", text)


__all__ = ["normalize_text_cached", "tokenize_cached", "strip_punctuation_cached"]
