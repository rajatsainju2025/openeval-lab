from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

from ..core import Adapter
from ..tools.base import Tool


@dataclass
class AgentStep:
    """A single step in an agent's trajectory."""

    thought: Optional[str]
    action: Optional[str]
    input: Optional[str]
    observation: Optional[str]


@dataclass
class AgentResult:
    success: bool
    final_answer: Any
    steps: List[AgentStep]
    error: Optional[str] = None


class Agent(ABC):
    """Abstract base class for multi-step, tool-using agents."""

    name: str

    def __init__(self, adapter: Adapter, tools: Optional[List[Tool]] = None):
        self.adapter = adapter
        self.tools = {t.name: t for t in (tools or [])}

    def get_tool(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    @abstractmethod
    def run(
        self, task_input: Any, **kwargs: Any
    ) -> AgentResult:  # pragma: no cover - interface only
        raise NotImplementedError
