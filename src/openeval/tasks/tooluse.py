from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Mapping, Optional

from ..core import Task, Example
from ..agents.base import Agent


@dataclass
class ToolUseExample(Example):
    tools: Optional[List[str]] = None  # allowed tool names for this example


class ToolUseTask(Task):
    name = "tool_use"

    def __init__(self, agent: Agent, prompt_template: Optional[str] = None):
        super().__init__(prompt_template)
        self.agent = agent

    def build_prompt(self, ex: Example) -> str:
        # Not used for agentic tasks; agent consumes raw input
        return str(ex.input)

    def postprocess(self, raw_output: str) -> Any:
        # Not used; agent returns structured result
        return raw_output

    def evaluate(self, *args, **kwargs):  # pragma: no cover - custom loop
        raise NotImplementedError("Use ToolUseRunner to evaluate agentic tasks")
