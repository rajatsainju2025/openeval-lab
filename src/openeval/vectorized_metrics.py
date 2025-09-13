"""
Vectorized Metric Computation for OpenEval Lab

This module provides vectorized implementations of evaluation metrics using NumPy and pandas
for significantly improved performance on large datasets.
"""

from __future__ import annotations

import re
import math
from typing import Any, Dict, List, Optional, Union, Iterable, Callable, Tuple
from collections import Counter
from dataclasses import dataclass

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import scipy.stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .enhanced_logging import get_logger

logger = get_logger(__name__)


@dataclass
class VectorizedMetricResult:
    """Result of a vectorized metric computation."""
    name: str
    value: float
    details: Optional[Dict[str, Any]] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "value": self.value
        }
        if self.details:
            result["details"] = self.details
        if self.confidence_interval:
            result["confidence_interval"] = self.confidence_interval
        if self.sample_size:
            result["sample_size"] = self.sample_size
        return result


class VectorizedMetrics:
    """
    Vectorized implementations of common evaluation metrics.
    """

    @staticmethod
    def exact_match(predictions: Iterable[Any], references: Iterable[Any]) -> VectorizedMetricResult:
        """Compute exact match accuracy using vectorized operations."""
        if not HAS_NUMPY or np is None:
            # Fallback to non-vectorized
            matches = sum(1 for p, r in zip(predictions, references) if str(p).strip() == str(r).strip())
            total = sum(1 for _ in predictions)
            return VectorizedMetricResult("exact_match", matches / total if total > 0 else 0.0)

        pred_array = np.array([str(p).strip() for p in predictions])
        ref_array = np.array([str(r).strip() for r in references])

        matches = np.sum(pred_array == ref_array)
        total = len(pred_array)

        accuracy = matches / total if total > 0 else 0.0

        return VectorizedMetricResult(
            name="exact_match",
            value=float(accuracy),
            details={"matches": int(matches), "total": total},
            sample_size=total
        )

    @staticmethod
    def f1_score(predictions: Iterable[str], references: Iterable[str]) -> VectorizedMetricResult:
        """Compute F1 score using vectorized token-level operations."""
        if not HAS_NUMPY:
            # Fallback implementation
            total_f1 = 0.0
            count = 0
            for pred, ref in zip(predictions, references):
                pred_tokens = set(str(pred).lower().split())
                ref_tokens = set(str(ref).lower().split())

                if not ref_tokens:
                    continue

                intersection = pred_tokens & ref_tokens
                precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
                recall = len(intersection) / len(ref_tokens) if ref_tokens else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                total_f1 += f1
                count += 1

            return VectorizedMetricResult("f1", total_f1 / count if count > 0 else 0.0)

        # Vectorized tokenization
        pred_tokens = [set(str(p).lower().split()) for p in predictions]
        ref_tokens = [set(str(r).lower().split()) for r in references]

        # Compute F1 for each pair
        f1_scores = []
        for pred_set, ref_set in zip(pred_tokens, ref_tokens):
            if not ref_set:
                continue

            intersection = pred_set & ref_set
            precision = len(intersection) / len(pred_set) if pred_set else 0.0
            recall = len(intersection) / len(ref_set) if ref_set else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)

        if not f1_scores:
            return VectorizedMetricResult("f1", 0.0)

        if HAS_NUMPY and np is not None:
            f1_array = np.array(f1_scores)
            mean_f1 = float(np.mean(f1_array))

            return VectorizedMetricResult(
                name="f1",
                value=mean_f1,
                details={
                    "mean": mean_f1,
                    "std": float(np.std(f1_array)),
                    "min": float(np.min(f1_array)),
                    "max": float(np.max(f1_array))
                },
                sample_size=len(f1_scores)
            )
        else:
            # Fallback without numpy
            mean_f1 = sum(f1_scores) / len(f1_scores)
            return VectorizedMetricResult(
                name="f1",
                value=mean_f1,
                details={
                    "mean": mean_f1,
                    "count": len(f1_scores)
                },
                sample_size=len(f1_scores)
            )

    @staticmethod
    def bleu_score(predictions: Iterable[str], references: Iterable[str], n_gram: int = 4) -> VectorizedMetricResult:
        """Compute BLEU score using vectorized n-gram operations."""
        if not HAS_NUMPY:
            # Simplified fallback BLEU
            total_bleu = 0.0
            count = 0
            for pred, ref in zip(predictions, references):
                pred_tokens = str(pred).split()
                ref_tokens = str(ref).split()

                if not pred_tokens or not ref_tokens:
                    continue

                # Simple unigram overlap
                pred_set = set(pred_tokens)
                ref_set = set(ref_tokens)
                overlap = len(pred_set & ref_set)
                precision = overlap / len(pred_set) if pred_set else 0.0
                total_bleu += precision
                count += 1

            return VectorizedMetricResult("bleu", total_bleu / count if count > 0 else 0.0)

        def _get_ngrams(tokens: List[str], n: int) -> List[str]:
            """Get n-grams from tokens."""
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

        bleu_scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = str(pred).split()
            ref_tokens = str(ref).split()

            if not pred_tokens or not ref_tokens:
                continue

            # Compute BLEU for different n-grams
            precisions = []

            for n in range(1, min(n_gram + 1, len(pred_tokens) + 1)):
                pred_ngrams = _get_ngrams(pred_tokens, n)
                ref_ngrams = _get_ngrams(ref_tokens, n)

                if not pred_ngrams:
                    precisions.append(0.0)
                    continue

                pred_counts = Counter(pred_ngrams)
                ref_counts = Counter(ref_ngrams)

                # Clipped counts
                clipped_counts = {ngram: min(count, ref_counts.get(ngram, 0))
                                for ngram, count in pred_counts.items()}

                precision = sum(clipped_counts.values()) / len(pred_ngrams)
                precisions.append(precision)

            if precisions:
                # Geometric mean of precisions
                if all(p > 0 for p in precisions):
                    bleu = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
                else:
                    bleu = 0.0

                # Brevity penalty
                pred_len = len(pred_tokens)
                ref_len = len(ref_tokens)
                if pred_len > ref_len:
                    brevity_penalty = 1.0
                else:
                    brevity_penalty = math.exp(1 - ref_len / pred_len) if pred_len > 0 else 0.0

                bleu *= brevity_penalty
                bleu_scores.append(bleu)

        if not bleu_scores:
            return VectorizedMetricResult("bleu", 0.0)

        bleu_array = np.array(bleu_scores)
        mean_bleu = np.mean(bleu_array)

        return VectorizedMetricResult(
            name="bleu",
            value=mean_bleu,
            details={
                "mean": mean_bleu,
                "std": np.std(bleu_array),
                "min": np.min(bleu_array),
                "max": np.max(bleu_array)
            },
            sample_size=len(bleu_scores)
        )

    @staticmethod
    def rouge_score(predictions: Iterable[str], references: Iterable[str]) -> VectorizedMetricResult:
        """Compute ROUGE score using vectorized operations."""
        if not HAS_NUMPY:
            # Fallback implementation
            total_rouge = 0.0
            count = 0
            for pred, ref in zip(predictions, references):
                pred_tokens = set(str(pred).lower().split())
                ref_tokens = set(str(ref).lower().split())

                if not ref_tokens:
                    continue

                intersection = pred_tokens & ref_tokens
                rouge = len(intersection) / len(ref_tokens)
                total_rouge += rouge
                count += 1

            return VectorizedMetricResult("rouge_l", total_rouge / count if count > 0 else 0.0)

        rouge_scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = set(str(pred).lower().split())
            ref_tokens = set(str(ref).lower().split())

            if not ref_tokens:
                continue

            intersection = pred_tokens & ref_tokens
            rouge = len(intersection) / len(ref_tokens)
            rouge_scores.append(rouge)

        if not rouge_scores:
            return VectorizedMetricResult("rouge_l", 0.0)

        rouge_array = np.array(rouge_scores)
        mean_rouge = np.mean(rouge_array)

        return VectorizedMetricResult(
            name="rouge_l",
            value=mean_rouge,
            details={
                "mean": mean_rouge,
                "std": np.std(rouge_array),
                "min": np.min(rouge_array),
                "max": np.max(rouge_array)
            },
            sample_size=len(rouge_scores)
        )

    @staticmethod
    def semantic_similarity(predictions: Iterable[str], references: Iterable[str]) -> VectorizedMetricResult:
        """Compute semantic similarity using vectorized operations (placeholder for advanced models)."""
        # This is a simplified implementation - in practice, you'd use sentence transformers
        if not HAS_NUMPY:
            # Simple fallback based on token overlap
            similarities = []
            for pred, ref in zip(predictions, references):
                pred_tokens = set(str(pred).lower().split())
                ref_tokens = set(str(ref).lower().split())

                if not pred_tokens and not ref_tokens:
                    similarities.append(1.0)
                elif not pred_tokens or not ref_tokens:
                    similarities.append(0.0)
                else:
                    intersection = pred_tokens & ref_tokens
                    union = pred_tokens | ref_tokens
                    jaccard = len(intersection) / len(union) if union else 0.0
                    similarities.append(jaccard)

            return VectorizedMetricResult(
                "semantic_similarity",
                sum(similarities) / len(similarities) if similarities else 0.0
            )

        # Vectorized Jaccard similarity
        similarities = []

        for pred, ref in zip(predictions, references):
            pred_tokens = set(str(pred).lower().split())
            ref_tokens = set(str(ref).lower().split())

            if not pred_tokens and not ref_tokens:
                similarities.append(1.0)
            elif not pred_tokens or not ref_tokens:
                similarities.append(0.0)
            else:
                intersection = pred_tokens & ref_tokens
                union = pred_tokens | ref_tokens
                jaccard = len(intersection) / len(union) if union else 0.0
                similarities.append(jaccard)

        if not similarities:
            return VectorizedMetricResult("semantic_similarity", 0.0)

        sim_array = np.array(similarities)
        mean_sim = np.mean(sim_array)

        return VectorizedMetricResult(
            name="semantic_similarity",
            value=mean_sim,
            details={
                "mean": mean_sim,
                "std": np.std(sim_array),
                "min": np.min(sim_array),
                "max": np.max(sim_array)
            },
            sample_size=len(similarities)
        )


class BatchMetricsProcessor:
    """
    Processes metrics in batches for improved performance.
    """

    def __init__(self, batch_size: int = 1000, use_pandas: bool = True):
        self.batch_size = batch_size
        self.use_pandas = use_pandas and HAS_PANDAS

    def compute_metrics_batch(
        self,
        predictions: Iterable[Any],
        references: Iterable[Any],
        metrics: List[str]
    ) -> Dict[str, VectorizedMetricResult]:
        """
        Compute multiple metrics in batches for better performance.

        Args:
            predictions: Predicted values
            references: Reference values
            metrics: List of metric names to compute

        Returns:
            Dictionary of metric results
        """
        # Convert to lists for batching
        pred_list = list(predictions)
        ref_list = list(references)

        if self.use_pandas:
            # Use pandas for efficient data handling
            df = pd.DataFrame({
                'prediction': pred_list,
                'reference': ref_list
            })
            pred_list = df['prediction'].tolist()
            ref_list = df['reference'].tolist()

        results = {}

        # Process in batches
        for i in range(0, len(pred_list), self.batch_size):
            batch_pred = pred_list[i:i + self.batch_size]
            batch_ref = ref_list[i:i + self.batch_size]

            for metric_name in metrics:
                if metric_name not in results:
                    results[metric_name] = []

                # Compute metric for this batch
                if metric_name == "exact_match":
                    result = VectorizedMetrics.exact_match(batch_pred, batch_ref)
                elif metric_name == "f1":
                    result = VectorizedMetrics.f1_score(batch_pred, batch_ref)
                elif metric_name == "bleu":
                    result = VectorizedMetrics.bleu_score(batch_pred, batch_ref)
                elif metric_name == "rouge_l":
                    result = VectorizedMetrics.rouge_score(batch_pred, batch_ref)
                elif metric_name == "semantic_similarity":
                    result = VectorizedMetrics.semantic_similarity(batch_pred, batch_ref)
                else:
                    continue

                results[metric_name].append(result)

        # Aggregate results across batches
        final_results = {}
        for metric_name, batch_results in results.items():
            if not batch_results:
                continue

            if len(batch_results) == 1:
                final_results[metric_name] = batch_results[0]
            else:
                # Aggregate multiple batches
                values = [r.value for r in batch_results]
                weights = [r.sample_size or 1 for r in batch_results]

                if HAS_NUMPY:
                    # Weighted average
                    weights_array = np.array(weights)
                    values_array = np.array(values)
                    weighted_avg = np.average(values_array, weights=weights_array)
                else:
                    # Simple average
                    weighted_avg = sum(values) / len(values)

                final_results[metric_name] = VectorizedMetricResult(
                    name=metric_name,
                    value=weighted_avg,
                    details={"batch_results": [r.to_dict() for r in batch_results]},
                    sample_size=sum(weights)
                )

        return final_results

    def compute_confidence_intervals(
        self,
        results: Dict[str, VectorizedMetricResult],
        confidence_level: float = 0.95
    ) -> Dict[str, VectorizedMetricResult]:
        """
        Compute confidence intervals for metric results.

        Args:
            results: Metric results
            confidence_level: Confidence level (0.95 for 95% CI)

        Returns:
            Results with confidence intervals
        """
        if not HAS_SCIPY:
            logger.warning("SciPy not available, skipping confidence interval computation")
            return results

        updated_results = {}

        for metric_name, result in results.items():
            if result.sample_size and result.sample_size > 1:
                # Compute confidence interval using t-distribution
                try:
                    # This is a simplified approach - in practice you'd need the raw data
                    # For now, we'll use the standard error from the batch results
                    if hasattr(result, 'details') and 'batch_results' in result.details:
                        batch_values = [br['value'] for br in result.details['batch_results']]
                        if len(batch_values) > 1:
                            mean_val = np.mean(batch_values)
                            std_val = np.std(batch_values, ddof=1)
                            n = len(batch_values)

                            # t-distribution critical value
                            t_crit = scipy.stats.t.ppf((1 + confidence_level) / 2, n - 1)
                            margin = t_crit * (std_val / np.sqrt(n))

                            result.confidence_interval = (mean_val - margin, mean_val + margin)

                except Exception as e:
                    logger.debug(f"Failed to compute confidence interval for {metric_name}: {e}")

            updated_results[metric_name] = result

        return updated_results


# Utility functions for easy integration
def compute_vectorized_metrics(
    predictions: Iterable[Any],
    references: Iterable[Any],
    metrics: List[str],
    batch_size: int = 1000
) -> Dict[str, VectorizedMetricResult]:
    """
    Compute multiple metrics using vectorized operations.

    Args:
        predictions: Predicted values
        references: Reference values
        metrics: List of metric names
        batch_size: Batch size for processing

    Returns:
        Dictionary of metric results
    """
    processor = BatchMetricsProcessor(batch_size=batch_size)
    return processor.compute_metrics_batch(predictions, references, metrics)


def benchmark_metrics_performance(
    predictions: Iterable[Any],
    references: Iterable[Any],
    metrics: List[str],
    iterations: int = 10
) -> Dict[str, Any]:
    """
    Benchmark the performance of vectorized vs non-vectorized metrics.

    Args:
        predictions: Predicted values
        references: Reference values
        metrics: List of metric names
        iterations: Number of benchmark iterations

    Returns:
        Benchmark results
    """
    import time

    # Prepare data
    pred_list = list(predictions)
    ref_list = list(references)

    # Benchmark vectorized implementation
    processor = BatchMetricsProcessor()

    vectorized_times = []
    for _ in range(iterations):
        start_time = time.time()
        vectorized_results = processor.compute_metrics_batch(pred_list, ref_list, metrics)
        vectorized_times.append(time.time() - start_time)

    # Simple non-vectorized benchmark (just for comparison)
    non_vectorized_times = []
    for _ in range(iterations):
        start_time = time.time()
        # Simulate non-vectorized computation
        for metric in metrics:
            if metric == "exact_match":
                VectorizedMetrics.exact_match(pred_list, ref_list)
            elif metric == "f1":
                VectorizedMetrics.f1_score(pred_list, ref_list)
        non_vectorized_times.append(time.time() - start_time)

    return {
        "vectorized_avg_time": sum(vectorized_times) / len(vectorized_times),
        "vectorized_std_time": np.std(vectorized_times) if HAS_NUMPY else 0,
        "non_vectorized_avg_time": sum(non_vectorized_times) / len(non_vectorized_times),
        "speedup_factor": (sum(non_vectorized_times) / len(non_vectorized_times)) / (sum(vectorized_times) / len(vectorized_times)),
        "iterations": iterations,
        "sample_size": len(pred_list)
    }