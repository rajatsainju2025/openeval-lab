from __future__ import annotations

from dataclasses import dataclass

from ..core import Example, Task


@dataclass
class SummarizationTask(Task):
    name: str = "summarization"
    instruction: str = "Summarize the following text concisely."
    max_words: int | None = 30

    def build_prompt(self, ex: Example) -> str:
        suffix = f" (max {self.max_words} words)" if self.max_words else ""
        return f"{self.instruction}{suffix}\n\nText: {ex.input}\nSummary:"

    def postprocess(self, raw_output: str):
        return raw_output.strip().splitlines()[0]
