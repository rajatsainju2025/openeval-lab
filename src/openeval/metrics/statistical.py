"""Advanced evaluation metrics with statistical analysis."""

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import math
from dataclasses import dataclass

from ..core import Metric


@dataclass
class StatisticalResult:
    """Container for statistical test results."""
    metric_value: float
    confidence_interval: Tuple[float, float]
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    sample_size: int = 0


class BootstrapMetric(Metric):
    """Base class for metrics with bootstrap confidence intervals."""
    
    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95):
        """
        Initialize bootstrap metric.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals (0.95 = 95%)
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
    
    def bootstrap_sample(self, predictions: List[Any], references: List[Any]) -> Tuple[List[Any], List[Any]]:
        """Create a bootstrap sample."""
        import random
        n = len(predictions)
        indices = [random.randint(0, n-1) for _ in range(n)]
        
        boot_pred = [predictions[i] for i in indices]
        boot_ref = [references[i] for i in indices]
        
        return boot_pred, boot_ref
    
    def compute_base_metric(self, predictions: List[Any], references: List[Any]) -> float:
        """Compute the base metric value. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def compute_with_bootstrap(
        self, 
        predictions: Iterable[Any], 
        references: Iterable[Any]
    ) -> StatisticalResult:
        """Compute metric with bootstrap confidence intervals."""
        pred_list = list(predictions)
        ref_list = list(references)
        
        if len(pred_list) != len(ref_list):
            raise ValueError("Predictions and references must have same length")
        
        # Compute main metric
        main_score = self.compute_base_metric(pred_list, ref_list)
        
        # Bootstrap sampling
        bootstrap_scores = []
        for _ in range(self.n_bootstrap):
            boot_pred, boot_ref = self.bootstrap_sample(pred_list, ref_list)
            boot_score = self.compute_base_metric(boot_pred, boot_ref)
            bootstrap_scores.append(boot_score)
        
        # Compute confidence interval
        bootstrap_scores.sort()
        alpha = 1 - self.confidence_level
        lower_idx = int(alpha / 2 * self.n_bootstrap)
        upper_idx = int((1 - alpha / 2) * self.n_bootstrap)
        
        lower_bound = bootstrap_scores[lower_idx]
        upper_bound = bootstrap_scores[upper_idx]
        
        return StatisticalResult(
            metric_value=main_score,
            confidence_interval=(lower_bound, upper_bound),
            sample_size=len(pred_list)
        )


class BootstrapAccuracy(BootstrapMetric):
    """Accuracy metric with bootstrap confidence intervals."""
    
    name: str = "bootstrap_accuracy"
    
    def compute_base_metric(self, predictions: List[Any], references: List[Any]) -> float:
        """Compute accuracy."""
        correct = sum(1 for p, r in zip(predictions, references) if p == r)
        return correct / len(predictions) if predictions else 0.0
    
    def compute(self, predictions: Iterable[Any], references: Iterable[Any]) -> Mapping[str, float]:
        """Compute accuracy with bootstrap confidence intervals."""
        result = self.compute_with_bootstrap(predictions, references)
        
        return {
            "accuracy": result.metric_value,
            "accuracy_ci_lower": result.confidence_interval[0], 
            "accuracy_ci_upper": result.confidence_interval[1],
            "sample_size": float(result.sample_size),
        }


class PairedBootstrapTest(Metric):
    """Paired bootstrap significance test between two sets of predictions."""
    
    name: str = "paired_bootstrap_test"
    
    def __init__(self, n_bootstrap: int = 1000):
        self.n_bootstrap = n_bootstrap
    
    def compute_difference(
        self, 
        pred1: List[Any], 
        pred2: List[Any], 
        references: List[Any]
    ) -> float:
        """Compute difference in accuracy between two prediction sets."""
        acc1 = sum(1 for p, r in zip(pred1, references) if p == r) / len(references)
        acc2 = sum(1 for p, r in zip(pred2, references) if p == r) / len(references)
        return acc1 - acc2
    
    def paired_bootstrap_test(
        self,
        predictions1: List[Any],
        predictions2: List[Any], 
        references: List[Any]
    ) -> StatisticalResult:
        """Perform paired bootstrap significance test."""
        n = len(references)
        
        # Observed difference
        observed_diff = self.compute_difference(predictions1, predictions2, references)
        
        # Bootstrap sampling
        bootstrap_diffs = []
        for _ in range(self.n_bootstrap):
            # Sample with replacement
            import random
            indices = [random.randint(0, n-1) for _ in range(n)]
            
            boot_pred1 = [predictions1[i] for i in indices]
            boot_pred2 = [predictions2[i] for i in indices]
            boot_ref = [references[i] for i in indices]
            
            boot_diff = self.compute_difference(boot_pred1, boot_pred2, boot_ref)
            bootstrap_diffs.append(boot_diff)
        
        # Compute p-value (two-tailed test)
        extreme_count = sum(1 for d in bootstrap_diffs if abs(d) >= abs(observed_diff))
        p_value = extreme_count / self.n_bootstrap
        
        # Confidence interval
        bootstrap_diffs.sort()
        lower_idx = int(0.025 * self.n_bootstrap)
        upper_idx = int(0.975 * self.n_bootstrap)
        
        return StatisticalResult(
            metric_value=observed_diff,
            confidence_interval=(bootstrap_diffs[lower_idx], bootstrap_diffs[upper_idx]),
            p_value=p_value,
            sample_size=n
        )


class EffectSizeMetric(Metric):
    """Compute effect sizes for comparing model performance."""
    
    name: str = "effect_size"
    
    def cohens_d(self, scores1: List[float], scores2: List[float]) -> float:
        """Compute Cohen's d effect size."""
        if not scores1 or not scores2:
            return 0.0
        
        mean1 = sum(scores1) / len(scores1)
        mean2 = sum(scores2) / len(scores2)
        
        # Pooled standard deviation
        var1 = sum((x - mean1) ** 2 for x in scores1) / (len(scores1) - 1) if len(scores1) > 1 else 0
        var2 = sum((x - mean2) ** 2 for x in scores2) / (len(scores2) - 1) if len(scores2) > 1 else 0
        
        pooled_std = math.sqrt(((len(scores1) - 1) * var1 + (len(scores2) - 1) * var2) / 
                              (len(scores1) + len(scores2) - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def compute(self, predictions: Iterable[Any], references: Iterable[Any]) -> Mapping[str, float]:
        """Placeholder - effect size requires two sets of predictions."""
        return {"cohens_d": 0.0}


class McNemar(Metric):
    """McNemar's test for comparing two binary classifiers."""
    
    name: str = "mcnemar_test"
    
    def mcnemar_test(
        self, 
        pred1: List[Any], 
        pred2: List[Any], 
        references: List[Any]
    ) -> StatisticalResult:
        """Perform McNemar's test."""
        # Create contingency table
        correct1 = [p == r for p, r in zip(pred1, references)]
        correct2 = [p == r for p, r in zip(pred2, references)]
        
        # Contingency table: both correct, 1 correct 2 wrong, 1 wrong 2 correct, both wrong
        both_correct = sum(1 for c1, c2 in zip(correct1, correct2) if c1 and c2)
        only1_correct = sum(1 for c1, c2 in zip(correct1, correct2) if c1 and not c2)
        only2_correct = sum(1 for c1, c2 in zip(correct1, correct2) if not c1 and c2)
        both_wrong = sum(1 for c1, c2 in zip(correct1, correct2) if not c1 and not c2)
        
        # McNemar's statistic
        if only1_correct + only2_correct == 0:
            chi_square = 0.0
            p_value = 1.0
        else:
            chi_square = (abs(only1_correct - only2_correct) - 1) ** 2 / (only1_correct + only2_correct)
            # Approximate p-value using chi-square distribution with df=1
            p_value = 1.0 - self._chi_square_cdf(chi_square, df=1)
        
        return StatisticalResult(
            metric_value=chi_square,
            confidence_interval=(0.0, chi_square),  # Not meaningful for chi-square
            p_value=p_value,
            sample_size=len(references)
        )
    
    def _chi_square_cdf(self, x: float, df: int) -> float:
        """Approximate chi-square CDF for df=1."""
        if x <= 0:
            return 0.0
        # Simple approximation for df=1
        return 2 * (1 - math.exp(-x/2) * (1 + x/2))
    
    def compute(self, predictions: Iterable[Any], references: Iterable[Any]) -> Mapping[str, float]:
        """Placeholder - McNemar requires two prediction sets."""
        return {"mcnemar_statistic": 0.0, "p_value": 1.0}


class PearsonCorrelation(Metric):
    """Pearson correlation coefficient for continuous predictions."""
    
    name: str = "pearson_correlation"
    
    def compute(self, predictions: Iterable[float], references: Iterable[float]) -> Mapping[str, float]:
        """Compute Pearson correlation coefficient."""
        pred_list = list(predictions)
        ref_list = list(references)
        
        if len(pred_list) != len(ref_list) or len(pred_list) == 0:
            return {"pearson_r": 0.0, "p_value": 1.0}
        
        # Compute means
        pred_mean = sum(pred_list) / len(pred_list)
        ref_mean = sum(ref_list) / len(ref_list)
        
        # Compute correlation coefficient
        numerator = sum((p - pred_mean) * (r - ref_mean) for p, r in zip(pred_list, ref_list))
        pred_sq = sum((p - pred_mean) ** 2 for p in pred_list)
        ref_sq = sum((r - ref_mean) ** 2 for r in ref_list)
        
        denominator = math.sqrt(pred_sq * ref_sq)
        
        if denominator == 0:
            return {"pearson_r": 0.0, "p_value": 1.0}
        
        r = numerator / denominator
        
        # Approximate p-value using t-distribution
        n = len(pred_list)
        if n <= 2:
            p_value = 1.0
        elif abs(r) >= 0.999:  # Near perfect correlation
            p_value = 0.001  # Very significant
        else:
            t_stat = r * math.sqrt((n - 2) / (1 - r**2))
            # Simple approximation for p-value
            p_value = max(0.0, min(1.0, 2 * (1 - abs(t_stat) / (abs(t_stat) + math.sqrt(n - 2)))))
        
        return {
            "pearson_r": r,
            "p_value": p_value,
            "sample_size": float(n)
        }
