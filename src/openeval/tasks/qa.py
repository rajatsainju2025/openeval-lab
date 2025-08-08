from __future__ import annotations

from dataclasses import dataclass

from ..core import Example, Task


@dataclass
class QATask(Task):
    name: str = "qa"
    instruction: str = "Answer the question concisely."

    def build_prompt(self, ex: Example) -> str:
        return f"{self.instruction}\n\nQ: {ex.input}\nA:"

    def postprocess(self, raw_output: str):
        return raw_output.strip().splitlines()[0]
