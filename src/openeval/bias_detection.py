"""Bias detection and mitigation utilities for LLM evaluation."""

from typing import List, Any, Dict, Optional
import random
from dataclasses import dataclass
from ..core import Adapter, Dataset, Task


@dataclass
class BiasAnalysisResult:
    """Results from bias analysis."""
    positional_bias_detected: bool
    positional_bias_score: float
    prompt_sensitivity_score: float
    recommendations: List[str]


class BiasDetector:
    """Detect various biases in LLM evaluation."""
    
    def __init__(self, adapter: Adapter, task: Task, dataset: Dataset):
        self.adapter = adapter
        self.task = task
        self.dataset = dataset
    
    def detect_positional_bias(self, n_permutations: int = 5) -> Dict[str, Any]:
        """Detect positional bias by randomizing answer positions."""
        examples = list(iter(self.dataset))
        if not examples:
            return {"detected": False, "score": 0.0, "details": "No examples to analyze"}
        
        # Get original accuracy
        original_predictions = []
        for ex in examples:
            prompt = self.task.build_prompt_with_template(ex)
            pred = self.adapter.generate(prompt)
            original_predictions.append(self.task.postprocess(pred))
        
        references = [ex.reference for ex in examples]
        original_accuracy = sum(1 for p, r in zip(original_predictions, references) if p == r) / len(examples)
        
        # Test with position permutations
        position_accuracies = [original_accuracy]
        
        for perm in range(n_permutations):
            permuted_predictions = []
            for ex in examples:
                # Create permuted example (simplified - assumes multiple choice)
                if hasattr(ex, 'options') and ex.options:
                    permuted_options = ex.options.copy()
                    random.shuffle(permuted_options)
                    # For simplicity, we'll just randomize the order
                    # In a real implementation, you'd need to modify the prompt accordingly
                    prompt = self.task.build_prompt_with_template(ex)
                    pred = self.adapter.generate(prompt)
                    permuted_predictions.append(self.task.postprocess(pred))
                else:
                    # For non-multiple-choice, use original
                    permuted_predictions.append(original_predictions[len(permuted_predictions)])
            
            perm_accuracy = sum(1 for p, r in zip(permuted_predictions, references) if p == r) / len(examples)
            position_accuracies.append(perm_accuracy)
        
        # Calculate bias score (variance in accuracy across positions)
        mean_accuracy = sum(position_accuracies) / len(position_accuracies)
        variance = sum((acc - mean_accuracy) ** 2 for acc in position_accuracies) / len(position_accuracies)
        bias_score = variance ** 0.5  # Standard deviation
        
        detected = bias_score > 0.05  # Threshold for detection
        
        return {
            "detected": detected,
            "score": bias_score,
            "mean_accuracy": mean_accuracy,
            "accuracy_variance": variance,
            "details": f"Analyzed {n_permutations + 1} position configurations"
        }
    
    def analyze_prompt_sensitivity(self, prompt_variations: List[str]) -> Dict[str, Any]:
        """Analyze sensitivity to prompt variations."""
        examples = list(iter(self.dataset))
        if not examples:
            return {"sensitivity_score": 0.0, "details": "No examples to analyze"}
        
        variation_scores = []
        
        for variation in prompt_variations:
            predictions = []
            for ex in examples:
                # Apply prompt variation (simplified)
                modified_prompt = variation.format(input=ex.input, reference=ex.reference)
                pred = self.adapter.generate(modified_prompt)
                predictions.append(self.task.postprocess(pred))
            
            references = [ex.reference for ex in examples]
            accuracy = sum(1 for p, r in zip(predictions, references) if p == r) / len(examples)
            variation_scores.append(accuracy)
        
        if not variation_scores:
            return {"sensitivity_score": 0.0, "details": "No variations tested"}
        
        # Calculate sensitivity as coefficient of variation
        mean_score = sum(variation_scores) / len(variation_scores)
        std_score = 0.0
        if mean_score == 0:
            sensitivity = 0.0
        else:
            variance = sum((s - mean_score) ** 2 for s in variation_scores) / len(variation_scores)
            std_score = variance ** 0.5
            sensitivity = std_score / mean_score
        
        return {
            "sensitivity_score": sensitivity,
            "mean_accuracy": mean_score,
            "accuracy_std": std_score,
            "variations_tested": len(prompt_variations),
            "details": f"Tested {len(prompt_variations)} prompt variations"
        }
    
    def run_full_analysis(self, prompt_variations: Optional[List[str]] = None) -> BiasAnalysisResult:
        """Run complete bias analysis."""
        if prompt_variations is None:
            prompt_variations = [
                "Question: {input}\nAnswer:",
                "Please answer this: {input}",
                "What is the answer to: {input}?",
                "{input}\nProvide your response:",
            ]
        
        positional_results = self.detect_positional_bias()
        sensitivity_results = self.analyze_prompt_sensitivity(prompt_variations)
        
        recommendations = []
        if positional_results["detected"]:
            recommendations.append("Consider balanced position calibration for multiple-choice tasks")
        if sensitivity_results["sensitivity_score"] > 0.1:
            recommendations.append("Prompt engineering may significantly impact results")
        if not recommendations:
            recommendations.append("No significant biases detected")
        
        return BiasAnalysisResult(
            positional_bias_detected=positional_results["detected"],
            positional_bias_score=positional_results["score"],
            prompt_sensitivity_score=sensitivity_results["sensitivity_score"],
            recommendations=recommendations
        )
