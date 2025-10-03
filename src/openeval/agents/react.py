from __future__ import annotations

from typing import Any, List

from .base import Agent, AgentResult, AgentStep
from ..core import Adapter
from ..tools.base import Tool


class ReActAgent(Agent):
    name = "react"

    def __init__(self, adapter: Adapter, tools: List[Tool]):
        super().__init__(adapter, tools)

    def run(self, task_input: Any, **kwargs: Any) -> AgentResult:
        # Minimal ReAct-like loop: Think -> Act (one tool) -> Observe -> Answer
        steps: List[AgentStep] = []
        try:
            # Ask the LLM which tool to use and what to compute (very simplified)
            prompt = (
                "You are an agent. Decide on one tool and input to solve the problem.\n"
                f"Problem: {task_input}\n"
                "Respond strictly as: TOOL_NAME: <name>\nTOOL_INPUT: <input>\n"
            )
            plan = self.adapter.generate(prompt)
            tool_name = None
            tool_input = None
            for line in plan.splitlines():
                if line.upper().startswith("TOOL_NAME:"):
                    tool_name = line.split(":", 1)[1].strip()
                if line.upper().startswith("TOOL_INPUT:"):
                    tool_input = line.split(":", 1)[1].strip()
            steps.append(AgentStep(thought=plan, action=tool_name, input=tool_input, observation=None))

            if not tool_name or not tool_input or tool_name not in self.tools:
                # Fallback: answer directly
                answer = self.adapter.generate(f"Answer the problem: {task_input}")
                steps.append(AgentStep(thought=None, action=None, input=None, observation=answer))
                return AgentResult(True, answer, steps)

            tool = self.tools[tool_name]
            tool_res = tool.run(tool_input)
            observation = str(tool_res.output if tool_res.success else tool_res.error)
            steps.append(AgentStep(thought=None, action=tool_name, input=tool_input, observation=observation))

            # Ask LLM to form final answer
            final = self.adapter.generate(
                "Given the problem and observation, provide the final answer succinctly.\n"
                f"Problem: {task_input}\nObservation: {observation}\nAnswer:"
            )
            return AgentResult(True, final, steps)
        except Exception as e:  # pragma: no cover - defensive
            return AgentResult(False, None, steps, str(e))
