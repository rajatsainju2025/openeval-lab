from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re

from ..core import Task, Example


class AgentTask(Task):
    """Task for evaluating AI agents with multi-step reasoning and tool use."""

    def __init__(
        self,
        instruction: str = "Solve this problem using available tools and reasoning.",
        max_steps: int = 10,
        tools_available: Optional[List[str]] = None,
        environment_setup: Optional[Dict[str, Any]] = None,
        success_criteria: Optional[Dict[str, Any]] = None,
        prompt_template: Optional[str] = None,
    ):
        """Initialize agent task."""
        self.max_steps = max_steps
        self.tools_available = tools_available or ["search", "calculator", "code_execution"]
        self.environment_setup = environment_setup or {}
        self.success_criteria = success_criteria or {}

        if prompt_template is None:
            prompt_template = """{{instruction}}

Available tools: {{tools_available}}
Maximum steps: {{max_steps}}

Problem: {{input}}

Think step by step and use tools when needed. Provide your final answer clearly."""

        super().__init__(prompt_template)
        self.instruction = instruction

    def validate_record(self, record: Dict[str, Any]) -> bool:
        """Validate agent task record format."""
        required_fields = ["input", "reference"]
        return all(field in record for field in required_fields)

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Process agent task record with tool-augmented reasoning."""
        # Use template rendering with agent-specific variables

        example = Example(
            id=record.get("id", ""),
            input=record["input"],
            reference=record["reference"],
            meta={
                "max_steps": self.max_steps,
                "tools_available": ", ".join(self.tools_available),
                "instruction": self.instruction,
            },
        )

        prompt = self.build_prompt_with_template(example)

        processed = {
            "input": record["input"],
            "reference": record["reference"],
            "prompt": prompt,
            "max_steps": self.max_steps,
            "tools_available": self.tools_available,
            "environment": self.environment_setup,
            "success_criteria": self.success_criteria,
        }

        return processed


@dataclass
class AgentTrajectory:
    """Represents a single agent execution trajectory."""

    steps: List[Dict[str, Any]]
    final_answer: str
    total_steps: int
    tools_used: List[str]
    execution_time: float
    success: bool
    reasoning_trace: List[str]

    @classmethod
    def from_prediction(cls, prediction: str, max_steps: int = 10) -> AgentTrajectory:
        """Parse agent prediction into trajectory format."""
        steps = []
        tools_used = []
        reasoning_trace = []

        # Parse prediction for tool usage and reasoning
        lines = prediction.split("\n")
        current_step = 0
        current_reasoning = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect tool usage
            tool_match = re.search(r"Tool:\s*(\w+)", line, re.IGNORECASE)
            if tool_match:
                tool_name = tool_match.group(1).lower()
                tools_used.append(tool_name)
                steps.append(
                    {
                        "step": current_step,
                        "tool": tool_name,
                        "reasoning": " ".join(current_reasoning),
                    }
                )
                current_reasoning = []
                current_step += 1

            # Detect reasoning
            elif any(
                keyword in line.lower() for keyword in ["think", "reason", "consider", "analyze"]
            ):
                current_reasoning.append(line)

            # Detect final answer
            elif "final answer" in line.lower() or "answer:" in line.lower():
                final_answer = line.split(":", 1)[-1].strip() if ":" in line else line
                break
        else:
            # If no final answer found, use last line
            final_answer = lines[-1] if lines else ""

        return cls(
            steps=steps,
            final_answer=final_answer,
            total_steps=current_step,
            tools_used=list(set(tools_used)),
            execution_time=0.0,  # Will be set by evaluator
            success=False,  # Will be determined by metrics
            reasoning_trace=reasoning_trace,
        )


@dataclass
class AgentMetrics:
    """Metrics for evaluating agent performance."""

    @staticmethod
    def tool_usage_efficiency(trajectory: AgentTrajectory, optimal_tools: List[str]) -> float:
        """Measure efficiency of tool usage."""
        if not trajectory.tools_used:
            return 0.0

        # Calculate precision and recall for tool usage
        used_tools = set(trajectory.tools_used)
        optimal_set = set(optimal_tools)

        if not optimal_set:
            return 1.0 if not used_tools else 0.5  # Neutral score if no optimal tools specified

        precision = len(used_tools & optimal_set) / len(used_tools) if used_tools else 0.0
        recall = len(used_tools & optimal_set) / len(optimal_set) if optimal_set else 1.0

        return (precision + recall) / 2.0

    @staticmethod
    def reasoning_quality(trajectory: AgentTrajectory) -> float:
        """Evaluate quality of reasoning trace."""
        if not trajectory.reasoning_trace:
            return 0.0

        # Simple heuristics for reasoning quality
        scores = []
        for reasoning in trajectory.reasoning_trace:
            score = 0.0

            # Check for analytical keywords
            analytical_keywords = ["because", "therefore", "however", "although", "since"]
            score += min(0.3, len([k for k in analytical_keywords if k in reasoning.lower()]) * 0.1)

            # Check for evidence-based reasoning
            if any(
                word in reasoning.lower()
                for word in ["evidence", "data", "information", "based on"]
            ):
                score += 0.2

            # Check for conclusion
            if any(word in reasoning.lower() for word in ["conclude", "therefore", "thus", "so"]):
                score += 0.2

            # Length appropriateness (not too short, not too long)
            word_count = len(reasoning.split())
            if 10 <= word_count <= 100:
                score += 0.3
            elif word_count < 10:
                score += 0.1

            scores.append(min(1.0, score))

        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def task_completion_success(trajectory: AgentTrajectory, reference: str) -> float:
        """Evaluate if the agent successfully completed the task."""
        if not trajectory.final_answer:
            return 0.0

        # Simple string matching for success (can be enhanced with semantic similarity)
        answer_clean = trajectory.final_answer.lower().strip()
        reference_clean = reference.lower().strip()

        # Exact match
        if answer_clean == reference_clean:
            return 1.0

        # Partial match
        if reference_clean in answer_clean or answer_clean in reference_clean:
            return 0.8

        # Keyword overlap
        answer_words = set(answer_clean.split())
        reference_words = set(reference_clean.split())

        if answer_words & reference_words:
            overlap = len(answer_words & reference_words)
            total = len(answer_words | reference_words)
            return overlap / total if total > 0 else 0.0

        return 0.0

    @staticmethod
    def efficiency_score(trajectory: AgentTrajectory, max_steps: int) -> float:
        """Measure efficiency based on steps used vs. maximum allowed."""
        if trajectory.total_steps == 0:
            return 0.0

        # Penalize excessive steps, reward optimal step usage
        step_ratio = trajectory.total_steps / max_steps
        if step_ratio <= 0.5:
            return 1.0  # Very efficient
        elif step_ratio <= 0.8:
            return 0.8  # Good efficiency
        elif step_ratio <= 1.0:
            return 0.6  # Acceptable
        else:
            return max(0.0, 1.0 - (step_ratio - 1.0) * 0.5)  # Decreasing score for overtime
