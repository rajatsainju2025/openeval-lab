from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..core import Metric


@dataclass
class ToolExecutionMetric(Metric):
    name: str = "tool_execution"

    def compute(self, predictions: Iterable[Any], references: Iterable[Any]) -> Mapping[str, float]:
        # For agent tasks, predictions may be AgentResult and references strings
        total = 0
        correct = 0
        tool_calls = 0
        for pred, ref in zip(predictions, references):
            total += 1
            # Accept plain-string prediction too
            final = getattr(pred, "final_answer", pred)
            if isinstance(final, str) and isinstance(ref, str) and final.strip() == ref.strip():
                correct += 1
            steps = getattr(pred, "steps", []) or []
            tool_calls += sum(1 for s in steps if s.action)
        acc = correct / total if total else 0.0
        avg_tool_calls = tool_calls / total if total else 0.0
        return {"accuracy": acc, "avg_tool_calls": avg_tool_calls}
