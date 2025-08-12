"""
Log-likelihood evaluation adapter for multiple-choice tasks.

This module provides log-likelihood evaluation capabilities,
inspired by lm-evaluation-harness multiple-choice evaluation.
"""

from typing import Any, Dict, List, Optional, Protocol, Tuple
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
            context: The context/prompt text
            continuation: The continuation to evaluate
            
        Returns:
            Log-likelihood score (higher is better)
        """
        ...
    
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text (for compatibility with base Adapter)."""
        ...


class MultipleChoiceTask(Task):
    """Task for multiple-choice evaluation using log-likelihood."""
    
    name: str = "multiple_choice"
    
    def __init__(
        self,
        prompt_template: Optional[str] = None,
        normalize_length: bool = True,
        choice_prefix: str = "",
        choice_suffix: str = "",
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
        def build_context(self, ex: Example) -> str:
        """Build context for log-likelihood evaluation."""
        if isinstance(ex, MCQAExample):
            return f"Question: {ex.question}
Answer:"
        else:
            # Fallback for generic examples
            return str(ex.input)
    
    def get_choices(self, ex: Example) -> List[str]:
        """Extract choices from example."""
        if hasattr(ex, 'choices'):
            return ex.choices
        elif hasattr(ex.meta, 'choices') if ex.meta else False:
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
        best_choice = choices[best_idx]
        best_label = labels[best_idx]
        
        return {
            "prediction": best_label,  # Return label (A, B, C, D)
            "prediction_text": best_choice,  # Return full text
            "scores": scores,
            "choices": choices,
            "labels": labels,
            "context": context,
        }
    
    def postprocess(self, raw_output: str) -> Any:
        """For compatibility - not used in log-likelihood evaluation."""
        return raw_output.strip()


class MockLogLikelihoodAdapter:
    """Mock adapter for testing log-likelihood evaluation."""
    
    name: str = "mock_loglik"
    
    def __init__(self, choice_preferences: Optional[Dict[str, float]] = None):
        """
        Initialize mock adapter.
        
        Args:
            choice_preferences: Map choice text to preference scores
        """
        self.choice_preferences = choice_preferences or {}
    
    def loglikelihood(self, context: str, continuation: str) -> float:
        """Mock log-likelihood computation."""
        # Simple mock: prefer certain choices
        continuation_clean = continuation.strip()
        
        if continuation_clean in self.choice_preferences:
            return self.choice_preferences[continuation_clean]
        
        # Default: random-ish score based on text
        return -len(continuation_clean) * 0.1 + hash(continuation_clean) % 1000 / 1000.0
    
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Mock generation - return first preferred choice."""
        if self.choice_preferences:
            best_choice = max(self.choice_preferences.items(), key=lambda x: x[1])[0]
            return best_choice
        return "A"


def evaluate_multiple_choice_loglik(
    task: MultipleChoiceTask,
    adapter: LogLikelihoodAdapter,
    examples: List[Example],
) -> Dict[str, Any]:
    """
    Evaluate multiple-choice task using log-likelihood.
    
    Args:
        task: MultipleChoiceTask instance
        adapter: LogLikelihoodAdapter instance
        examples: List of examples to evaluate
        
    Returns:
        Evaluation results with accuracy and detailed predictions
    """
    predictions = []
    references = []
    detailed_results = []
    
    for ex in examples:
        result = task.predict(adapter, ex)
        prediction = result["prediction"]
        reference = ex.reference
        
        predictions.append(prediction)
        references.append(reference)
        detailed_results.append({
            **result,
            "example_id": ex.id,
            "reference": reference,
            "correct": prediction == reference,
        })
    
    # Compute accuracy
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    accuracy = correct / len(predictions) if predictions else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(predictions),
        "predictions": predictions,
        "references": references,
        "detailed_results": detailed_results,
    }
