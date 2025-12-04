from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ToolResult:
    """Result of a tool execution.

    Attributes:
        success: Whether the tool execution succeeded
        output: The result data (can be any type)
        error: Error message if execution failed
        latency_ms: Execution time in milliseconds
    """

    success: bool
    output: Any
    error: Optional[str] = None
    latency_ms: Optional[float] = None


class Tool(ABC):
    """Abstract base class for tools that agents can use.

    Tools provide discrete capabilities (e.g., web search, calculator, file I/O)
    that can be invoked by agents during task execution.

    Subclasses must implement:
        - name: Unique identifier for the tool
        - description: Human-readable explanation of what the tool does
        - run(): Execute the tool with given inputs

    Example:
        >>> class MyTool(Tool):
        ...     name = "my_tool"
        ...     description = "Does something useful"
        ...
        ...     def run(self, query: str, **kwargs: Any) -> ToolResult:
        ...         result = process(query)
        ...         return ToolResult(success=True, output=result)

    See src/openeval/tools/examples.py for complete implementation examples.
    """

    name: str
    description: str

    @abstractmethod
    def run(self, query: str, **kwargs: Any) -> ToolResult:
        """Execute the tool with the provided query and kwargs.

        Args:
            query: Primary input to the tool (interpretation varies by tool)
            **kwargs: Additional parameters specific to the tool

        Returns:
            ToolResult containing success status, output, and optional error/timing

        Example:
            >>> tool = Calculator()
            >>> result = tool.run("2 + 2")
            >>> assert result.success and result.output == 4
        """
        raise NotImplementedError
