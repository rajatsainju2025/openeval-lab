from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable
import numpy as np

try:
    import scipy.stats as stats
except ImportError:
    stats = None

try:
    from sklearn.metrics import brier_score_loss
except ImportError:
    brier_score_loss = None

from ..core import Metric


@dataclass
class UncertaintyMetrics:
    """Collection of uncertainty quantification metrics for LLM evaluation."""

    @staticmethod
    def expected_calibration_error(predictions: List[str], references: List[str],
                                 confidences: List[float], num_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        ECE measures the difference between predicted confidence and actual accuracy.
        Lower values indicate better calibration.
        """
        if len(predictions) != len(references) or len(predictions) != len(confidences):
            raise ValueError("Predictions, references, and confidences must have same length")

        # Create bins based on confidence
        bins = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(confidences, bins) - 1

        ece = 0.0
        total_samples = len(predictions)

        for bin_idx in range(num_bins):
            # Get samples in this bin
            bin_mask = bin_indices == bin_idx
            bin_size = np.sum(bin_mask)

            if bin_size == 0:
                continue

            # Calculate accuracy in this bin
            bin_predictions = [p for p, m in zip(predictions, bin_mask) if m]
            bin_references = [r for r, m in zip(references, bin_mask) if m]
            bin_confidences = [c for c, m in zip(confidences, bin_mask) if m]

            # Simple accuracy calculation (exact match)
            accuracy = np.mean([p.strip().lower() == r.strip().lower()
                              for p, r in zip(bin_predictions, bin_references)])

            # Average confidence in this bin
            avg_confidence = np.mean(bin_confidences)

            # Add to ECE
            ece += (bin_size / total_samples) * abs(avg_confidence - accuracy)

        return ece

    @staticmethod
    def brier_score(predictions: List[str], references: List[str],
                   probabilities: List[Dict[str, float]]) -> float:
        """
        Calculate Brier Score for probabilistic predictions.

        Measures the accuracy of probabilistic predictions.
        Lower values indicate better calibration.
        """
        if len(predictions) != len(references) or len(predictions) != len(probabilities):
            raise ValueError("All inputs must have same length")

        # Convert to binary classification format
        # This is a simplified version - in practice, you'd need proper probability distributions
        brier_scores = []

        for pred, ref, probs in zip(predictions, references, probabilities):
            # For simplicity, assume binary correct/incorrect
            correct = 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0

            # Use confidence as probability of being correct
            confidence = probs.get('confidence', 0.5)
            predicted_prob = confidence

            # Brier score for binary case: (predicted_prob - actual)^2
            brier = (predicted_prob - correct) ** 2
            brier_scores.append(brier)

        return float(np.mean(brier_scores))

    @staticmethod
    def confidence_interval_width(confidences: List[float], z_score: float = 1.96) -> float:
        """
        Calculate average width of confidence intervals.

        For a given confidence level (default 95%), this measures how wide
        the uncertainty intervals are on average.
        """
        if not confidences:
            return 0.0

        # Assuming confidences represent the width or can be used to estimate it
        # In practice, this would depend on how confidence is measured
        return float(np.mean(confidences))

    @staticmethod
    def uncertainty_correlation(predictions: List[str], references: List[str],
                              uncertainties: List[float]) -> float:
        """
        Calculate correlation between uncertainty and error likelihood.

        Positive correlation indicates that higher uncertainty predicts errors better.
        """
        if len(predictions) != len(references) or len(predictions) != len(uncertainties):
            raise ValueError("All inputs must have same length")

        # Calculate errors (1 for incorrect, 0 for correct)
        errors = [1.0 if p.strip().lower() != r.strip().lower() else 0.0
                 for p, r in zip(predictions, references)]

        # Calculate Pearson correlation
        if len(set(errors)) <= 1 or len(set(uncertainties)) <= 1:
            return 0.0  # No variation

        if stats is not None:
            correlation, _ = stats.pearsonr(errors, uncertainties)
            return float(correlation)
        else:
            # Fallback: simple correlation calculation
            return float(np.corrcoef(errors, uncertainties)[0, 1])

    @staticmethod
    def sharpness(uncertainties: List[float]) -> float:
        """
        Calculate sharpness of uncertainty estimates.

        Sharpness measures how concentrated the uncertainty distribution is.
        Lower values indicate sharper (more concentrated) distributions.
        """
        if not uncertainties:
            return 0.0

        return float(np.mean(uncertainties))

    @staticmethod
    def entropy_uncertainty(predictions: List[str], references: List[str],
                          logits: Optional[List[Dict[str, float]]] = None) -> float:
        """
        Calculate entropy-based uncertainty.

        Uses predictive entropy as a measure of uncertainty.
        Higher entropy indicates more uncertainty.
        """
        if not logits:
            # Fallback: use simple confidence-based uncertainty
            return float(np.mean([1.0 - (len(p.split()) / 100.0) for p in predictions]))

        entropies = []
        for logit_dict in logits:
            # Convert logits to probabilities
            logits_array = np.array(list(logit_dict.values()))
            exp_logits = np.exp(logits_array - np.max(logits_array))  # Numerical stability
            probs = exp_logits / np.sum(exp_logits)

            # Calculate entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))  # Add small epsilon to avoid log(0)
            entropies.append(entropy)

        return float(np.mean(entropies))

    @staticmethod
    def aleatoric_epistemic_decomposition(
        predictions: List[str],
        references: List[str],
        ensemble_predictions: Optional[List[List[str]]] = None
    ) -> Tuple[float, float]:
        """
        Decompose uncertainty into aleatoric (data) and epistemic (model) components.

        Returns (aleatoric_uncertainty, epistemic_uncertainty)
        """
        if not ensemble_predictions:
            # Cannot decompose without ensemble
            total_uncertainty = UncertaintyMetrics.entropy_uncertainty(predictions, references)
            return total_uncertainty * 0.5, total_uncertainty * 0.5

        aleatoric = []
        epistemic = []

        for i, ensemble_preds in enumerate(ensemble_predictions):
            # Aleatoric: average uncertainty across ensemble members
            member_entropies = []
            for pred in ensemble_preds:
                # Simplified: use prediction length as uncertainty proxy
                entropy = len(pred.split()) / 100.0
                member_entropies.append(entropy)

            aleatoric.append(np.mean(member_entropies))

            # Epistemic: uncertainty of ensemble mean
            mean_pred_length = np.mean([len(p.split()) for p in ensemble_preds])
            epistemic.append(abs(len(predictions[i].split()) - mean_pred_length) / 100.0)

        return float(np.mean(aleatoric)), float(np.mean(epistemic))


@dataclass
class UncertaintyQuantificationMetric(Metric):
    """Metric that incorporates uncertainty quantification."""

    name: str = "uncertainty_quantification"
    num_bins: int = 10

    def compute(self, predictions: Iterable[Any], references: Iterable[Any],
               **kwargs) -> Dict[str, float]:
        """Compute comprehensive uncertainty metrics."""
        # Convert to lists for processing
        pred_list = list(predictions)
        ref_list = list(references)

        # Extract uncertainty information from kwargs or predictions
        confidences = kwargs.get('confidences', [0.5] * len(pred_list))
        probabilities = kwargs.get('probabilities', [{}] * len(pred_list))
        logits = kwargs.get('logits', None)
        ensemble_predictions = kwargs.get('ensemble_predictions', None)

        results = {}

        # Expected Calibration Error
        try:
            results['ece'] = UncertaintyMetrics.expected_calibration_error(
                pred_list, ref_list, confidences, self.num_bins
            )
        except Exception:
            results['ece'] = 0.0

        # Brier Score
        try:
            results['brier_score'] = UncertaintyMetrics.brier_score(
                pred_list, ref_list, probabilities
            )
        except Exception:
            results['brier_score'] = 0.0

        # Confidence Interval Width
        results['confidence_interval_width'] = UncertaintyMetrics.confidence_interval_width(confidences)

        # Uncertainty-Error Correlation
        try:
            results['uncertainty_error_correlation'] = UncertaintyMetrics.uncertainty_correlation(
                pred_list, ref_list, confidences
            )
        except Exception:
            results['uncertainty_error_correlation'] = 0.0

        # Sharpness
        results['sharpness'] = UncertaintyMetrics.sharpness(confidences)

        # Entropy-based Uncertainty
        results['predictive_entropy'] = UncertaintyMetrics.entropy_uncertainty(
            pred_list, ref_list, logits
        )

        # Uncertainty Decomposition
        aleatoric, epistemic = UncertaintyMetrics.aleatoric_epistemic_decomposition(
            pred_list, ref_list, ensemble_predictions
        )
        results['aleatoric_uncertainty'] = aleatoric
        results['epistemic_uncertainty'] = epistemic

        return results
