from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    success: bool
    output: Any
    error: Optional[str] = None
    latency_ms: Optional[float] = None


class Tool(ABC):
    """Abstract base class for tools that agents can use."""

    name: str
    description: str

    @abstractmethod
    def run(self, query: str, **kwargs: Any) -> ToolResult:  # pragma: no cover - interface only
        """Execute the tool with the provided query and kwargs."""
        raise NotImplementedError
