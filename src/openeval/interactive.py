from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Callable
import json
import time
from pathlib import Path
import threading
import queue
from enum import Enum

from ..core import Task, Dataset, Adapter, Metric, Example


class InteractionMode(Enum):
    """Modes of human-AI interaction for evaluation."""
    ACTIVE_LEARNING = "active_learning"
    HUMAN_IN_LOOP = "human_in_loop"
    COLLABORATIVE = "collaborative"
    VALIDATION = "validation"


class FeedbackType(Enum):
    """Types of human feedback."""
    CORRECTION = "correction"
    CONFIDENCE = "confidence"
    PREFERENCE = "preference"
    EXPLANATION = "explanation"


@dataclass
class HumanFeedback:
    """Human feedback on a model prediction."""

    example_id: str
    original_prediction: str
    human_correction: Optional[str] = None
    confidence_score: Optional[float] = None
    feedback_type: FeedbackType = FeedbackType.CORRECTION
    explanation: Optional[str] = None
    timestamp: Optional[float] = None
    human_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class InteractiveSession:
    """An interactive evaluation session."""

    session_id: str
    task: Task
    dataset: Dataset
    adapter: Adapter
    metrics: List[Metric]
    mode: InteractionMode = InteractionMode.HUMAN_IN_LOOP
    max_iterations: int = 10
    convergence_threshold: float = 0.01

    # Session state
    current_iteration: int = 0
    feedback_history: Optional[List[HumanFeedback]] = None
    performance_history: Optional[List[Dict[str, float]]] = None
    active_examples: Optional[List[Example]] = None

    def __post_init__(self):
        if self.feedback_history is None:
            self.feedback_history = []
        if self.performance_history is None:
            self.performance_history = []
        if self.active_examples is None:
            self.active_examples = []

    def add_feedback(self, feedback: HumanFeedback) -> None:
        """Add human feedback to the session."""
        self.feedback_history.append(feedback)

    def get_next_examples(self, num_examples: int = 5) -> List[Example]:
        """Get next examples for human evaluation based on strategy."""
        if self.mode == InteractionMode.ACTIVE_LEARNING:
            return self._active_learning_selection(num_examples)
        elif self.mode == InteractionMode.HUMAN_IN_LOOP:
            return self._uncertainty_sampling(num_examples)
        else:
            # Default: random sampling
            import random
            all_examples = list(self.dataset)
            return random.sample(all_examples, min(num_examples, len(all_examples)))

    def _active_learning_selection(self, num_examples: int) -> List[Example]:
        """Select examples using active learning strategies."""
        # Simplified active learning: select examples with high uncertainty
        return self._uncertainty_sampling(num_examples)

    def _uncertainty_sampling(self, num_examples: int) -> List[Example]:
        """Select examples with highest model uncertainty."""
        examples_with_uncertainty = []

        for example in self.dataset:
            # Get model prediction and uncertainty
            prediction = self.adapter.predict([example.input])
            uncertainty = self._estimate_uncertainty(prediction[0] if prediction else "")

            examples_with_uncertainty.append((example, uncertainty))

        # Sort by uncertainty (highest first)
        examples_with_uncertainty.sort(key=lambda x: x[1], reverse=True)

        return [ex for ex, _ in examples_with_uncertainty[:num_examples]]

    def _estimate_uncertainty(self, prediction: str) -> float:
        """Estimate uncertainty of a prediction (simplified)."""
        # Simple heuristic: shorter predictions are more uncertain
        if not prediction:
            return 1.0

        # Normalize by reasonable prediction length
        length_score = min(len(prediction.split()) / 50.0, 1.0)
        return 1.0 - length_score  # Higher uncertainty for shorter predictions

    def update_model_with_feedback(self) -> Dict[str, float]:
        """Update model/adapter based on collected feedback."""
        if not self.feedback_history:
            return {}

        # In a real implementation, this would fine-tune the model
        # For now, we'll simulate improvement based on feedback

        feedback_quality = len(self.feedback_history) / (self.current_iteration + 1)
        improvement_factor = min(feedback_quality * 0.1, 0.05)  # Cap improvement

        # Simulate performance improvement
        if self.performance_history:
            last_performance = self.performance_history[-1]
            updated_performance = {}
            for metric_name, value in last_performance.items():
                # Simulate improvement for accuracy-like metrics
                if "accuracy" in metric_name.lower() or "exact_match" in metric_name.lower():
                    updated_performance[metric_name] = min(value + improvement_factor, 1.0)
                else:
                    updated_performance[metric_name] = value
        else:
            # Initial performance estimate
            updated_performance = {"accuracy": 0.7, "exact_match": 0.7}

        self.performance_history.append(updated_performance)
        self.current_iteration += 1

        return updated_performance

    def check_convergence(self) -> bool:
        """Check if the interactive session has converged."""
        if len(self.performance_history) < 2:
            return False

        # Check if performance improvement is below threshold
        recent = self.performance_history[-1]
        previous = self.performance_history[-2]

        max_change = 0.0
        for metric_name in recent:
            if metric_name in previous:
                change = abs(recent[metric_name] - previous[metric_name])
                max_change = max(max_change, change)

        return max_change < self.convergence_threshold

    def generate_feedback_summary(self) -> Dict[str, Any]:
        """Generate summary of feedback and session progress."""
        if not self.feedback_history:
            return {"total_feedback": 0, "feedback_types": {}, "session_progress": 0.0}

        feedback_types = {}
        for feedback in self.feedback_history:
            feedback_types[feedback.feedback_type.value] = feedback_types.get(feedback.feedback_type.value, 0) + 1

        progress = min(self.current_iteration / self.max_iterations, 1.0)

        return {
            "total_feedback": len(self.feedback_history),
            "feedback_types": feedback_types,
            "session_progress": progress,
            "iterations_completed": self.current_iteration,
            "has_converged": self.check_convergence(),
            "performance_trajectory": self.performance_history
        }


class InteractiveEvaluator:
    """Coordinator for interactive evaluation sessions."""

    def __init__(self):
        self.sessions: Dict[str, InteractiveSession] = {}
        self.feedback_queue: queue.Queue = queue.Queue()
        self.session_lock = threading.Lock()

    def create_session(self, task: Task, dataset: Dataset, adapter: Adapter,
                      metrics: List[Metric], mode: InteractionMode = InteractionMode.HUMAN_IN_LOOP) -> str:
        """Create a new interactive evaluation session."""
        import uuid
        session_id = str(uuid.uuid4())

        session = InteractiveSession(
            session_id=session_id,
            task=task,
            dataset=dataset,
            adapter=adapter,
            metrics=metrics,
            mode=mode
        )

        with self.session_lock:
            self.sessions[session_id] = session

        return session_id

    def get_session(self, session_id: str) -> Optional[InteractiveSession]:
        """Get an interactive session by ID."""
        with self.session_lock:
            return self.sessions.get(session_id)

    def submit_feedback(self, session_id: str, feedback: HumanFeedback) -> bool:
        """Submit human feedback to a session."""
        session = self.get_session(session_id)
        if not session:
            return False

        session.add_feedback(feedback)

        # Update model with feedback
        session.update_model_with_feedback()

        return True

    def get_next_batch(self, session_id: str, batch_size: int = 5) -> List[Example]:
        """Get next batch of examples for human evaluation."""
        session = self.get_session(session_id)
        if not session:
            return []

        return session.get_next_examples(batch_size)

    def run_interactive_evaluation(self, session_id: str) -> Dict[str, Any]:
        """Run complete interactive evaluation session."""
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}

        results = []

        for iteration in range(session.max_iterations):
            # Get next examples
            examples = session.get_next_examples(5)

            if not examples:
                break

            # In a real implementation, these would be presented to humans
            # For simulation, we'll auto-generate some feedback
            simulated_feedback = self._simulate_human_feedback(examples, session.adapter)

            for feedback in simulated_feedback:
                session.add_feedback(feedback)

            # Update model
            performance = session.update_model_with_feedback()
            results.append({
                "iteration": iteration,
                "examples_evaluated": len(examples),
                "feedback_collected": len(simulated_feedback),
                "performance": performance
            })

            # Check convergence
            if session.check_convergence():
                break

        return {
            "session_id": session_id,
            "total_iterations": len(results),
            "final_performance": session.performance_history[-1] if session.performance_history else {},
            "feedback_summary": session.generate_feedback_summary(),
            "iteration_results": results
        }

    def _simulate_human_feedback(self, examples: List[Example], adapter: Adapter) -> List[HumanFeedback]:
        """Simulate human feedback for examples (for demonstration)."""
        feedback = []

        for example in examples:
            # Get model prediction
            predictions = adapter.predict([example.input])
            prediction = predictions[0] if predictions else ""

            # Simulate human correction (simplified)
            if len(prediction.split()) < 3:  # Short predictions get corrections
                correction = f"Improved answer: {prediction} with additional context."
            else:
                correction = None

            # Simulate confidence score
            confidence = 0.8 if len(prediction.split()) > 5 else 0.5

            feedback.append(HumanFeedback(
                example_id=getattr(example, 'id', str(hash(example.input))),
                original_prediction=prediction,
                human_correction=correction,
                confidence_score=confidence,
                feedback_type=FeedbackType.CORRECTION,
                explanation="Simulated human feedback for demonstration"
            ))

        return feedback

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active interactive sessions."""
        with self.session_lock:
            return [
                {
                    "session_id": session_id,
                    "mode": session.mode.value,
                    "current_iteration": session.current_iteration,
                    "total_feedback": len(session.feedback_history),
                    "has_converged": session.check_convergence()
                }
                for session_id, session in self.sessions.items()
            ]

    def cleanup_session(self, session_id: str) -> bool:
        """Clean up a completed session."""
        with self.session_lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
        return False
