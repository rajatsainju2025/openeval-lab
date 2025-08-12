"""Log-likelihood accuracy metric for multiple-choice tasks."""

from typing import Any, Iterable, Mapping

from ..core import Metric


class LogLikelihoodAccuracy(Metric):
    """Accuracy metric for log-likelihood multiple-choice evaluation."""
    
    name: str = "loglik_accuracy"
    
    def compute(
        self, 
        predictions: Iterable[Any], 
        references: Iterable[Any]
    ) -> Mapping[str, float]:
        """
        Compute accuracy for multiple-choice predictions.
        
        Args:
            predictions: Predicted choice labels (A, B, C, D, etc.)
            references: Reference choice labels
            
        Returns:
            Dictionary with accuracy metrics
        """
        pred_list = list(predictions)
        ref_list = list(references)
        
        if not pred_list:
            return {"accuracy": 0.0, "correct": 0, "total": 0}
        
        correct = sum(1 for p, r in zip(pred_list, ref_list) if p == r)
        total = len(pred_list)
        accuracy = correct / total
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
