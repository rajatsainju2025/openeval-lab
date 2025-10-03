"""Calibration and uncertainty metrics."""

from typing import Any, Iterable, List, Mapping
import math
from dataclasses import dataclass

from ..core import Metric


@dataclass
class CalibrationError(Metric):
    """Expected Calibration Error (ECE) for binary classification confidence."""

    name: str = "calibration_error"
    n_bins: int = 10

    def compute(self, predictions: Iterable[Any], references: Iterable[Any]) -> Mapping[str, float]:
        """Compute Expected Calibration Error."""
        # This is a simplified implementation assuming predictions include confidence scores
        # In practice, you'd need confidence scores from the model
        pred_list = list(predictions)
        ref_list = list(references)

        if len(pred_list) != len(ref_list):
            return {"ece": 0.0, "error_code": -1.0}  # Use float codes for errors

        # For demonstration, assume binary classification with confidence
        # In real usage, predictions should include confidence scores
        confidences = []
        accuracies = []

        for pred, ref in zip(pred_list, ref_list):
            # Extract confidence if available (simplified)
            if isinstance(pred, dict) and "confidence" in pred:
                conf = pred["confidence"]
                acc = 1.0 if pred.get("prediction") == ref else 0.0
            else:
                # Fallback: assume exact match gives confidence 1.0, else 0.8
                conf = 1.0 if str(pred).strip() == str(ref).strip() else 0.8
                acc = 1.0 if str(pred).strip() == str(ref).strip() else 0.0

            confidences.append(conf)
            accuracies.append(acc)

        if not confidences:
            return {"ece": 0.0}

        # Compute ECE
        ece = self._compute_ece(confidences, accuracies, self.n_bins)

        return {"ece": ece, "n_bins": self.n_bins}

    def _compute_ece(self, confidences: List[float], accuracies: List[float], n_bins: int) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = [i / n_bins for i in range(n_bins + 1)]

        ece = 0.0
        for i in range(n_bins):
            bin_start = bin_boundaries[i]
            bin_end = bin_boundaries[i + 1]

            # Find samples in this confidence bin
            bin_indices = [
                j
                for j, conf in enumerate(confidences)
                if bin_start <= conf < bin_end or (i == n_bins - 1 and conf == 1.0)
            ]

            if not bin_indices:
                continue

            # Compute bin statistics
            bin_conf = sum(confidences[j] for j in bin_indices) / len(bin_indices)
            bin_acc = sum(accuracies[j] for j in bin_indices) / len(bin_indices)
            bin_size = len(bin_indices) / len(confidences)

            ece += bin_size * abs(bin_conf - bin_acc)

        return ece


@dataclass
class ConfidenceIntervals(Metric):
    """Compute confidence intervals for accuracy metrics."""

    name: str = "confidence_intervals"
    confidence_level: float = 0.95

    def compute(self, predictions: Iterable[Any], references: Iterable[Any]) -> Mapping[str, float]:
        """Compute confidence intervals using normal approximation."""
        pred_list = list(predictions)
        ref_list = list(references)

        if len(pred_list) != len(ref_list):
            return {"accuracy": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "error_code": -1.0}

        # Compute accuracy
        correct = sum(1 for p, r in zip(pred_list, ref_list) if str(p).strip() == str(r).strip())
        accuracy = correct / len(pred_list)

        # Compute standard error
        n = len(pred_list)
        if n == 0:
            return {"accuracy": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

        se = math.sqrt(accuracy * (1 - accuracy) / n)

        # Compute confidence interval
        z_score = 1.96  # For 95% confidence
        margin = z_score * se

        return {
            "accuracy": accuracy,
            "ci_lower": max(0.0, accuracy - margin),
            "ci_upper": min(1.0, accuracy + margin),
            "margin_of_error": margin,
            "confidence_level": self.confidence_level,
        }


@dataclass
class Perplexity(Metric):
    """Compute perplexity for language models."""

    name: str = "perplexity"

    def compute(self, predictions: Iterable[Any], references: Iterable[Any]) -> Mapping[str, float]:
        """Compute perplexity from log probabilities."""
        # This assumes predictions include log probabilities
        pred_list = list(predictions)

        total_log_prob = 0.0
        total_tokens = 0

        for pred in pred_list:
            if isinstance(pred, dict) and "log_probs" in pred:
                log_probs = pred["log_probs"]
                total_log_prob += sum(log_probs)
                total_tokens += len(log_probs)
            else:
                # Skip if no log probs available
                continue

        if total_tokens == 0:
            return {"perplexity": 0.0, "error_code": -1.0}

        avg_log_prob = total_log_prob / total_tokens
        perplexity = math.exp(-avg_log_prob)

        return {
            "perplexity": perplexity,
            "avg_log_prob": avg_log_prob,
            "total_tokens": total_tokens,
        }


@dataclass
class DiversityMetrics(Metric):
    """Compute diversity metrics for generated text."""

    name: str = "diversity"

    def compute(self, predictions: Iterable[Any], references: Iterable[Any]) -> Mapping[str, float]:
        """Compute diversity metrics."""
        pred_list = [str(p) for p in predictions]

        if not pred_list:
            return {"unique_ngrams": 0.0, "self_bleu": 0.0}

        # Compute unique n-grams (simplified)
        all_words = []
        for pred in pred_list:
            words = pred.lower().split()
            all_words.extend(words)

        if not all_words:
            return {"unique_ngrams": 0.0, "self_bleu": 0.0}

        # Unique unigrams
        unique_unigrams = len(set(all_words))
        total_unigrams = len(all_words)
        unique_ratio = unique_unigrams / total_unigrams if total_unigrams > 0 else 0.0

        # Self-BLEU (simplified approximation)
        # In practice, you'd compute BLEU between pairs of predictions
        self_bleu = 1.0 - unique_ratio  # Rough approximation

        return {
            "unique_unigrams_ratio": unique_ratio,
            "self_bleu": self_bleu,
            "total_words": total_unigrams,
            "unique_words": unique_unigrams,
        }
