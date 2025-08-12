"""Tests for log-likelihood multiple choice evaluation."""

import pytest
from pathlib import Path

from openeval.loglikelihood import (
    MCQATask,
    MockLogLikelihoodAdapter,
    evaluate_multiple_choice
)
from openeval.datasets.mcqa import MCQAExample
from openeval.core import Example


class TestMockLogLikelihoodAdapter:
    """Test the mock adapter for log-likelihood evaluation."""
    
    def test_basic_functionality(self):
        """Test basic mock adapter functionality."""
        adapter = MockLogLikelihoodAdapter()
        
        # Test log-likelihood computation
        score = adapter.loglikelihood("Question: What is 2+2?", " 4")
        assert isinstance(score, float)
        assert adapter.call_count == 1
        
        # Test predefined scores
        predefined_scores = {"test|answer": 5.0}
        adapter = MockLogLikelihoodAdapter(predefined_scores)
        
        score = adapter.loglikelihood("test", "answer")
        assert score == 5.0


class TestMCQATask:
    """Test MCQA task functionality."""
    
    def test_basic_mcqa_task(self):
        """Test basic MCQA task operations."""
        task = MCQATask()
        
        # Create an MCQA example
        example = MCQAExample(
            id="test1",
            input="What is the capital of France?",
            reference="A",
            question="What is the capital of France?",
            choices=["Paris", "London", "Berlin", "Madrid"],
            answer="A"
        )
        
        # Test choice extraction
        choices = task.get_choices(example)
        assert choices == ["Paris", "London", "Berlin", "Madrid"]
        
        # Test choice labels
        labels = task.get_choice_labels(choices)
        assert labels == ["A", "B", "C", "D"]
        
        # Test context building
        context = task.build_context(example)
        assert "What is the capital of France?" in context
        
    def test_mcqa_task_with_adapter(self):
        """Test MCQA task prediction with adapter."""
        # Create mock adapter with biased scores
        scores = {
            "Question: What is the capital of France?\nAnswer:| Paris": 1.0,
            "Question: What is the capital of France?\nAnswer:| London": 0.5,
            "Question: What is the capital of France?\nAnswer:| Berlin": 0.3,
            "Question: What is the capital of France?\nAnswer:| Madrid": 0.2
        }
        adapter = MockLogLikelihoodAdapter(scores)
        task = MCQATask()
        
        example = MCQAExample(
            id="test1",
            input="What is the capital of France?",
            reference="A",
            question="What is the capital of France?",
            choices=["Paris", "London", "Berlin", "Madrid"],
            answer="A"
        )
        
        result = task.predict(adapter, example)
        
        # Check prediction structure
        assert "prediction" in result
        assert "choice" in result
        assert "scores" in result
        assert "confidence" in result
        
        # Should predict Paris (highest score)
        assert result["prediction"] == "A"
        assert result["choice"] == "Paris"
        
    def test_length_normalization(self):
        """Test length normalization option."""
        # Without normalization
        task = MCQATask(normalize_length=False)
        
        # With normalization (default)
        task_norm = MCQATask(normalize_length=True)
        
        assert not task.normalize_length
        assert task_norm.normalize_length
        
    def test_choice_formatting(self):
        """Test choice prefix/suffix formatting."""
        task = MCQATask(choice_prefix=" (", choice_suffix=")")
        
        adapter = MockLogLikelihoodAdapter()
        example = MCQAExample(
            id="test1",
            input="Test question?",
            reference="A",
            question="Test question?",
            choices=["Answer A", "Answer B"],
            answer="A"
        )
        
        # Test that formatting is applied in evaluation
        score = task.evaluate_choice(adapter, "Context", "Answer A")
        assert isinstance(score, float)
        
        # The adapter should have been called with formatted continuation
        assert adapter.call_count == 1


class TestEvaluateMultipleChoice:
    """Test the evaluation function."""
    
    def test_evaluation_function(self):
        """Test the main evaluation function."""
        task = MCQATask()
        adapter = MockLogLikelihoodAdapter()
        
        examples = [
            MCQAExample(
                id="q1",
                input="What is 2+2?",
                reference="A",
                question="What is 2+2?",
                choices=["4", "5", "6"],
                answer="A"
            ),
            MCQAExample(
                id="q2",
                input="What is 3+3?",
                reference="B",
                question="What is 3+3?",
                choices=["5", "6", "7"],
                answer="B"
            )
        ]
        
        result = evaluate_multiple_choice(task, adapter, examples)
        
        # Check result structure
        assert "predictions" in result
        assert "num_examples" in result
        assert "avg_confidence" in result
        assert "adapter" in result
        
        assert result["num_examples"] == 2
        assert result["adapter"] == "mock_loglik"
        assert len(result["predictions"]) == 2
        
        # Check prediction structure
        pred = result["predictions"][0]
        assert "id" in pred
        assert "prediction" in pred
        assert "reference" in pred
        assert "scores" in pred


class TestGenericExampleSupport:
    """Test support for generic examples (fallback behavior)."""
    
    def test_generic_example_support(self):
        """Test that task works with generic examples."""
        task = MCQATask()
        
        # Create a generic example without MCQA-specific attributes
        example = Example(
            id="generic1",
            input="What is the best color?",
            reference="Blue",
            meta={"choices": ["Red", "Blue", "Green"]}
        )
        
        # Should extract choices from meta
        choices = task.get_choices(example)
        assert choices == ["Red", "Blue", "Green"]
        
        # Should build context from input
        context = task.build_context(example)
        assert "What is the best color?" in context
        
    def test_fallback_choices(self):
        """Test fallback to binary choices."""
        task = MCQATask()
        
        example = Example(
            id="binary1",
            input="Is this correct?",
            reference="Yes"
        )
        
        # Should fall back to binary choice
        choices = task.get_choices(example)
        assert choices == ["No", "Yes"]
