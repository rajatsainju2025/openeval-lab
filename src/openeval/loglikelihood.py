"""
Log-likelihood evaluation adapter for multiple-choice tasks.

This module provides log-likelihood evaluation capabilities,
inspired by lm-evaluation-harness multiple-choice evaluation.
"""

from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
import math
import numpy as np

from .core import Adapter, Example, Task
from .datasets.mcqa import MCQAExample


class LogLikelihoodAdapter(Protocol):
    """Adapter that supports log-likelihood computation."""
    
    name: str
    
    def loglikelihood(self, context: str, continuation: str) -> float:
        """
        Compute log-likelihood of continuation given context.
        
        Args:
            context: The context/prompt
            continuation: The continuation to evaluate
            
        Returns:
            Log-likelihood score (higher is better)
        """
        ...


class MockLogLikelihoodAdapter:
    """Mock adapter for testing log-likelihood functionality."""
    
    name = "mock_loglik"
    
    def __init__(self, scores: Optional[Dict[str, float]] = None):
        """Initialize with predefined scores for testing."""
        self.scores = scores or {}
        self.call_count = 0
    
    def loglikelihood(self, context: str, continuation: str) -> float:
        """Return mock log-likelihood scores."""
        self.call_count += 1
        
        # Use predefined scores if available
        key = f"{context}|{continuation}"
        if key in self.scores:
            return self.scores[key]
        
        # Default scoring: prefer shorter, common continuations
        base_score = -len(continuation) * 0.1
        
        # Boost score for common answer patterns
        continuation = continuation.strip().lower()
        if continuation in ['a', 'yes', 'true', 'correct']:
            base_score += 2.0
        elif continuation in ['b', 'no', 'false', 'incorrect']:
            base_score += 1.0
            
        return base_score


class MultipleChoiceTask(Task):
    """Base task for multiple-choice evaluation using log-likelihood."""
    
    def __init__(self, prompt_template: Optional[str] = None):
        super().__init__(prompt_template)
    
    def build_prompt(self, ex: Example) -> str:
        """Build the prompt for evaluation."""
        raise NotImplementedError
    
    def predict(self, adapter: LogLikelihoodAdapter, ex: Example) -> Dict[str, Any]:
        """Predict using log-likelihood evaluation."""
        raise NotImplementedError


class MCQATask(MultipleChoiceTask):
    """Multiple choice question answering task using log-likelihood."""
    
    def __init__(
        self,
        prompt_template: Optional[str] = None,
        normalize_length: bool = True,
        choice_prefix: str = " ",
        choice_suffix: str = ""
    ):
        """
        Initialize multiple-choice task.
        
        Args:
            prompt_template: Jinja2 template for prompt
            normalize_length: Whether to normalize by token length
            choice_prefix: Prefix for each choice (e.g., " ")
            choice_suffix: Suffix for each choice (e.g., "")
        """
        super().__init__(prompt_template)
        self.normalize_length = normalize_length
        self.choice_prefix = choice_prefix
        self.choice_suffix = choice_suffix
    
    def build_prompt(self, ex: Example) -> str:
        """Build context prompt (without choices)."""
        return self.build_context(ex)
        
    def build_context(self, ex: Example) -> str:
        """Build context for log-likelihood evaluation."""
        if isinstance(ex, MCQAExample):
            return f"Question: {ex.question}\nAnswer:"
        else:
            # Fallback for generic examples
            return str(ex.input)
    
    def get_choices(self, ex: Example) -> List[str]:
        """Extract choices from example."""
        if isinstance(ex, MCQAExample):
            return ex.choices
        elif ex.meta and 'choices' in ex.meta:
            return ex.meta['choices']
        else:
            # Fallback: assume binary choice
            return ["No", "Yes"]
    
    def get_choice_labels(self, choices: List[str]) -> List[str]:
        """Get choice labels (A, B, C, D, ...)."""
        return [chr(ord('A') + i) for i in range(len(choices))]
    
    def evaluate_choice(
        self,
        adapter: LogLikelihoodAdapter,
        context: str,
        choice: str,
    ) -> float:
        """Evaluate a single choice using log-likelihood."""
        continuation = f"{self.choice_prefix}{choice}{self.choice_suffix}"
        loglik = adapter.loglikelihood(context, continuation)
        
        if self.normalize_length:
            # Normalize by token length (approximate with character length)
            # This helps shorter answers not be unfairly penalized
            length = max(1, len(continuation.strip()))
            return loglik / length
        
        return loglik
    
    def predict(self, adapter: LogLikelihoodAdapter, ex: Example) -> Dict[str, Any]:
        """
        Predict the best choice using log-likelihood.
        
        Returns:
            Dictionary with prediction, scores, and metadata
        """
        context = self.build_prompt_with_template(ex)
        choices = self.get_choices(ex)
        labels = self.get_choice_labels(choices)
        
        # Evaluate each choice
        scores = []
        for choice in choices:
            score = self.evaluate_choice(adapter, context, choice)
            scores.append(score)
        
        # Find best choice
        best_idx = np.argmax(scores)
        best_label = labels[best_idx]
        best_choice = choices[best_idx]
        
        return {
            "prediction": best_label,
            "choice": best_choice,
            "scores": dict(zip(labels, scores)),
            "confidence": scores[best_idx] - np.mean(scores),
            "choice_distribution": self._softmax(scores)
        }
    
    def _softmax(self, scores: List[float]) -> List[float]:
        """Apply softmax to get probability distribution."""
        exp_scores = np.exp(np.array(scores) - np.max(scores))
        return (exp_scores / np.sum(exp_scores)).tolist()
    
    def postprocess(self, raw_output: str) -> Any:
        """Post-process the prediction (no-op for log-likelihood)."""
        return raw_output


def evaluate_multiple_choice(
    task: MultipleChoiceTask,
    adapter: LogLikelihoodAdapter,
    examples: List[Union[Example, MCQAExample]],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate multiple-choice task using log-likelihood.
    
    Args:
        task: The multiple-choice task
        adapter: Adapter supporting log-likelihood
        examples: List of examples to evaluate
        verbose: Whether to print progress
        
    Returns:
        Evaluation results with predictions and metadata
    """
    predictions = []
    total_score = 0.0
    
    for i, ex in enumerate(examples):
        if verbose:
            print(f"Evaluating example {i+1}/{len(examples)}...")
        
        result = task.predict(adapter, ex)
        predictions.append({
            "id": ex.id,
            "prediction": result["prediction"],
            "choice": result["choice"],
            "reference": getattr(ex, 'answer', ex.reference),
            "scores": result["scores"],
            "confidence": result["confidence"]
        })
        
        total_score += result["confidence"]
    
    return {
        "predictions": predictions,
        "num_examples": len(examples),
        "avg_confidence": total_score / len(examples) if examples else 0.0,
        "adapter": adapter.name
    }
