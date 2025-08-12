"""Tests for statistical evaluation metrics."""

import pytest
from openeval.metrics.statistical import (
    BootstrapAccuracy,
    PairedBootstrapTest,
    EffectSizeMetric,
    McNemar,
    PearsonCorrelation,
    StatisticalResult,
)


class TestBootstrapAccuracy:
    """Test bootstrap accuracy metric."""
    
    def test_bootstrap_accuracy_basic(self):
        """Test basic bootstrap accuracy computation."""
        metric = BootstrapAccuracy(n_bootstrap=100)
        
        predictions = ["A", "B", "A", "A", "B"]
        references = ["A", "B", "A", "B", "A"]
        
        result = metric.compute(predictions, references)
        
        assert "accuracy" in result
        assert "accuracy_ci_lower" in result
        assert "accuracy_ci_upper" in result
        assert "sample_size" in result
        
        # Check accuracy calculation
        expected_accuracy = 3/5  # 3 correct out of 5
        assert result["accuracy"] == expected_accuracy
        assert result["sample_size"] == 5.0
        
        # CI bounds should be reasonable
        assert result["accuracy_ci_lower"] <= result["accuracy"]
        assert result["accuracy_ci_upper"] >= result["accuracy"]
    
    def test_bootstrap_accuracy_perfect(self):
        """Test bootstrap accuracy with perfect predictions."""
        metric = BootstrapAccuracy(n_bootstrap=50)
        
        predictions = ["A", "B", "C"]
        references = ["A", "B", "C"]
        
        result = metric.compute(predictions, references)
        
        assert result["accuracy"] == 1.0
        # With perfect accuracy, CI should be tight
        assert result["accuracy_ci_lower"] >= 0.5  # Should be high
        assert result["accuracy_ci_upper"] == 1.0
    
    def test_bootstrap_accuracy_empty(self):
        """Test bootstrap accuracy with empty input."""
        metric = BootstrapAccuracy(n_bootstrap=10)
        
        result = metric.compute([], [])
        
        assert result["accuracy"] == 0.0
        assert result["sample_size"] == 0.0


class TestPairedBootstrapTest:
    """Test paired bootstrap significance test."""
    
    def test_paired_test_basic(self):
        """Test basic paired bootstrap test."""
        metric = PairedBootstrapTest(n_bootstrap=100)
        
        pred1 = ["A", "B", "A", "A", "B"]
        pred2 = ["A", "A", "A", "B", "B"] 
        references = ["A", "B", "A", "B", "A"]
        
        result = metric.paired_bootstrap_test(pred1, pred2, references)
        
        assert isinstance(result, StatisticalResult)
        assert result.p_value is not None
        assert 0.0 <= result.p_value <= 1.0
        assert result.sample_size == 5
        
        # Check difference calculation
        acc1 = 3/5  # pred1 accuracy
        acc2 = 3/5  # pred2 accuracy  
        expected_diff = acc1 - acc2
        assert abs(result.metric_value - expected_diff) < 1e-10
    
    def test_paired_test_identical(self):
        """Test paired test with identical predictions."""
        metric = PairedBootstrapTest(n_bootstrap=50)
        
        predictions = ["A", "B", "A"]
        references = ["A", "B", "A"]
        
        result = metric.paired_bootstrap_test(predictions, predictions, references)
        
        # Identical predictions should have difference of 0
        assert result.metric_value == 0.0
        assert result.p_value >= 0.5  # Should not be significant


class TestEffectSizeMetric:
    """Test effect size calculations."""
    
    def test_cohens_d_basic(self):
        """Test Cohen's d calculation."""
        metric = EffectSizeMetric()
        
        scores1 = [0.8, 0.7, 0.9, 0.6, 0.8]
        scores2 = [0.6, 0.5, 0.7, 0.4, 0.6]
        
        cohens_d = metric.cohens_d(scores1, scores2)
        
        # Should be positive since scores1 > scores2
        assert cohens_d > 0
        assert isinstance(cohens_d, float)
    
    def test_cohens_d_identical(self):
        """Test Cohen's d with identical scores."""
        metric = EffectSizeMetric()
        
        scores = [0.7, 0.8, 0.6, 0.9]
        
        cohens_d = metric.cohens_d(scores, scores)
        
        # Identical distributions should have d = 0
        assert abs(cohens_d) < 1e-10
    
    def test_cohens_d_empty(self):
        """Test Cohen's d with empty inputs."""
        metric = EffectSizeMetric()
        
        cohens_d = metric.cohens_d([], [0.5, 0.6])
        assert cohens_d == 0.0
        
        cohens_d = metric.cohens_d([0.5], [])
        assert cohens_d == 0.0


class TestMcNemar:
    """Test McNemar's test."""
    
    def test_mcnemar_basic(self):
        """Test basic McNemar's test."""
        metric = McNemar()
        
        pred1 = ["A", "B", "A", "B", "A"]
        pred2 = ["A", "A", "B", "B", "A"]
        references = ["A", "B", "A", "B", "A"]
        
        result = metric.mcnemar_test(pred1, pred2, references)
        
        assert isinstance(result, StatisticalResult)
        assert result.p_value is not None
        assert 0.0 <= result.p_value <= 1.0
        assert result.sample_size == 5
        assert result.metric_value >= 0  # Chi-square is non-negative
    
    def test_mcnemar_identical(self):
        """Test McNemar with identical predictions."""
        metric = McNemar()
        
        predictions = ["A", "B", "A"]
        references = ["A", "B", "C"]
        
        result = metric.mcnemar_test(predictions, predictions, references)
        
        # Identical predictions should give chi-square = 0
        assert result.metric_value == 0.0
        assert result.p_value == 1.0


class TestPearsonCorrelation:
    """Test Pearson correlation coefficient."""
    
    def test_pearson_perfect_correlation(self):
        """Test perfect positive correlation."""
        metric = PearsonCorrelation()
        
        predictions = [1.0, 2.0, 3.0, 4.0, 5.0]
        references = [2.0, 4.0, 6.0, 8.0, 10.0]  # Perfect linear relationship
        
        result = metric.compute(predictions, references)
        
        assert "pearson_r" in result
        assert "p_value" in result
        assert "sample_size" in result
        
        # Should be perfect correlation
        assert abs(result["pearson_r"] - 1.0) < 1e-10
        assert result["sample_size"] == 5.0
        assert result["p_value"] < 0.1  # Should be significant
    
    def test_pearson_no_correlation(self):
        """Test no correlation."""
        metric = PearsonCorrelation()
        
        predictions = [1.0, 2.0, 3.0, 4.0]
        references = [2.0, 1.0, 4.0, 3.0]  # No clear relationship
        
        result = metric.compute(predictions, references)
        
        # Correlation should be weak
        assert abs(result["pearson_r"]) < 0.8
        assert result["sample_size"] == 4.0
    
    def test_pearson_negative_correlation(self):
        """Test negative correlation."""
        metric = PearsonCorrelation()
        
        predictions = [1.0, 2.0, 3.0, 4.0]
        references = [4.0, 3.0, 2.0, 1.0]  # Perfect negative relationship
        
        result = metric.compute(predictions, references)
        
        # Should be perfect negative correlation
        assert abs(result["pearson_r"] + 1.0) < 1e-10
    
    def test_pearson_constant_values(self):
        """Test correlation with constant values."""
        metric = PearsonCorrelation()
        
        predictions = [5.0, 5.0, 5.0]
        references = [1.0, 2.0, 3.0]
        
        result = metric.compute(predictions, references)
        
        # Constant predictions should give correlation = 0
        assert result["pearson_r"] == 0.0
        assert result["p_value"] == 1.0
    
    def test_pearson_empty(self):
        """Test correlation with empty inputs."""
        metric = PearsonCorrelation()
        
        result = metric.compute([], [])
        
        assert result["pearson_r"] == 0.0
        assert result["p_value"] == 1.0


class TestStatisticalResult:
    """Test StatisticalResult dataclass."""
    
    def test_statistical_result_creation(self):
        """Test creating StatisticalResult."""
        result = StatisticalResult(
            metric_value=0.85,
            confidence_interval=(0.75, 0.95),
            p_value=0.03,
            effect_size=0.4,
            sample_size=100
        )
        
        assert result.metric_value == 0.85
        assert result.confidence_interval == (0.75, 0.95)
        assert result.p_value == 0.03
        assert result.effect_size == 0.4
        assert result.sample_size == 100
    
    def test_statistical_result_defaults(self):
        """Test StatisticalResult with defaults."""
        result = StatisticalResult(
            metric_value=0.7,
            confidence_interval=(0.6, 0.8)
        )
        
        assert result.metric_value == 0.7
        assert result.confidence_interval == (0.6, 0.8)
        assert result.p_value is None
        assert result.effect_size is None
        assert result.sample_size == 0
