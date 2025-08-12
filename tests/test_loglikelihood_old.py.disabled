"""Tests for log-likelihood evaluation."""

import pytest

from openeval.loglikelihood import (
    MCQATask,
    MockLogLikelihoodAdapter,
    MultipleChoiceTask,
    evaluate_multiple_choice
)
from openeval.tasks.mcqa import MCQATask
from openeval.datasets.mcqa import MCQAExample
from openeval.metrics.loglik_accuracy import LogLikelihoodAccuracy


class TestMockLogLikelihoodAdapter:
    """Test mock adapter for log-likelihood."""
    
    def test_mock_adapter_basic(self):
        """Test basic mock adapter functionality."""
        adapter = MockLogLikelihoodAdapter()
        
        # Should return some score
        score = adapter.loglikelihood("What is 2+2?", " 4")
        assert isinstance(score, float)
        
        # Should generate something
        output = adapter.generate("Test prompt")
        assert isinstance(output, str)
    
    def test_mock_adapter_preferences(self):
        """Test mock adapter with preferences."""
        adapter = MockLogLikelihoodAdapter({
            " 4": 0.9,  # High preference for "4"
            " 5": 0.1,  # Low preference for "5"
        })
        
        score_4 = adapter.loglikelihood("What is 2+2?", " 4")
        score_5 = adapter.loglikelihood("What is 2+2?", " 5")
        
        assert score_4 > score_5  # Should prefer "4"


class TestMultipleChoiceTask:
    """Test multiple choice task functionality."""
    
    def test_basic_task(self):
        """Test basic multiple choice task."""
        task = MultipleChoiceTask()
        
        # Create test example
        example = MCQAExample(
            id="1",
            input="What is 2+2?",
            reference="B",
            question="What is 2+2?",
            choices=["3", "4", "5", "6"],
            answer="B"
        )
        
        # Test choice extraction
        choices = task.get_choices(example)
        assert choices == ["3", "4", "5", "6"]
        
        # Test choice labels
        labels = task.get_choice_labels(choices)
        assert labels == ["A", "B", "C", "D"]
    
    def test_prediction(self):
        """Test prediction with mock adapter."""
        task = MultipleChoiceTask()
        
        # Create adapter that prefers "4"
        adapter = MockLogLikelihoodAdapter({
            " 3": -1.0,
            " 4": 0.9,  # Highest score
            " 5": -0.5,
            " 6": -0.8,
        })
        
        example = MCQAExample(
            id="1",
            input="What is 2+2?",
            reference="B",
            question="What is 2+2?",
            choices=["3", "4", "5", "6"],
            answer="B"
        )
        
        result = task.predict(adapter, example)
        
        assert result["prediction"] == "B"  # Should pick "4" which is index 1 -> "B"
        assert result["prediction_text"] == "4"
        assert len(result["scores"]) == 4
        assert result["choices"] == ["3", "4", "5", "6"]


class TestMCQATask:
    """Test MCQA task specifically."""
    
    def test_mcqa_task_template(self):
        """Test MCQA task with Jinja template."""
        pytest.importorskip("jinja2")
        
        task = MCQATask()
        
        example = MCQAExample(
            id="1",
            input="What is 2+2?",
            reference="B",
            question="What is 2+2?",
            choices=["3", "4", "5", "6"],
            answer="B"
        )
        
        prompt = task.build_prompt(example)
        
        # Should contain the question and formatted choices
        assert "What is 2+2?" in prompt
        assert "A. 3" in prompt
        assert "B. 4" in prompt
        assert "C. 5" in prompt
        assert "D. 6" in prompt
        assert "Answer:" in prompt
    
    def test_mcqa_prediction(self):
        """Test MCQA task prediction."""
        task = MCQATask()
        
        # Create adapter that prefers "4"
        adapter = MockLogLikelihoodAdapter({
            " 3": -1.0,
            " 4": 0.9,  # Highest score
            " 5": -0.5,
            " 6": -0.8,
        })
        
        example = MCQAExample(
            id="1",
            input="What is 2+2?",
            reference="B",
            question="What is 2+2?",
            choices=["3", "4", "5", "6"],
            answer="B"
        )
        
        result = task.predict(adapter, example)
        assert result["prediction"] == "B"


class TestLogLikelihoodAccuracy:
    """Test log-likelihood accuracy metric."""
    
    def test_accuracy_computation(self):
        """Test accuracy computation."""
        metric = LogLikelihoodAccuracy()
        
        predictions = ["A", "B", "C", "A"]
        references = ["A", "B", "D", "A"]  # 3/4 correct
        
        result = metric.compute(predictions, references)
        
        assert result["accuracy"] == 0.75
        assert result["correct"] == 3
        assert result["total"] == 4
    
    def test_empty_predictions(self):
        """Test with empty predictions."""
        metric = LogLikelihoodAccuracy()
        
        result = metric.compute([], [])
        
        assert result["accuracy"] == 0.0
        assert result["correct"] == 0
        assert result["total"] == 0


class TestEvaluateMultipleChoice:
    """Test end-to-end multiple choice evaluation."""
    
    def test_full_evaluation(self):
        """Test full evaluation pipeline."""
        task = MCQATask()
        
        # Create adapter that prefers index 1 ("4" for 2+2, "Paris" for capital)
        adapter = MockLogLikelihoodAdapter({
            " 3": -1.0, " 4": 0.9, " 5": -0.5, " 6": -0.8,  # For math question
            " London": -0.5, " Paris": 0.8, " Berlin": -0.3, " Madrid": -0.6,  # For geography
        })
        
        examples = [
            MCQAExample(
                id="1",
                input="What is 2+2?",
                reference="B",
                question="What is 2+2?",
                choices=["3", "4", "5", "6"],
                answer="B"
            ),
            MCQAExample(
                id="2",
                input="What is the capital of France?",
                reference="B", 
                question="What is the capital of France?",
                choices=["London", "Paris", "Berlin", "Madrid"],
                answer="B"
            ),
        ]
        
        result = evaluate_multiple_choice(task, adapter, examples)
        
        assert result["accuracy"] == 1.0  # Both should be correct
        assert result["correct"] == 2
        assert result["total"] == 2
        assert len(result["detailed_results"]) == 2
