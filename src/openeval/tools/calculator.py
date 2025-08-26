from __future__ import annotations

import math
import time
from typing import Any

from .base import Tool, ToolResult


class Calculator(Tool):
    name = "calculator"
    description = "Evaluate simple arithmetic expressions using Python's eval with safe namespace."

    def run(self, query: str, **kwargs: Any) -> ToolResult:
        start = time.perf_counter()
        try:
            # Very limited safe eval context
            allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            allowed_names["abs"] = abs
            allowed_names["round"] = round
            result = eval(query, {"__builtins__": {}}, allowed_names)
            return ToolResult(True, result, None, (time.perf_counter() - start) * 1000)
        except Exception as e:  # pragma: no cover - defensive
            return ToolResult(False, None, str(e), (time.perf_counter() - start) * 1000)
