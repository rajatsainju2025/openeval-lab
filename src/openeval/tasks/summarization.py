from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core import Example, Task
from ..prompt import PromptTemplate


@dataclass
class SummarizationTask(Task):
    name: str = "summarization"
    instruction: str = "Summarize the following text concisely."
    max_words: int | None = 30
    prompt_template: Optional[PromptTemplate] = None

    def build_prompt(self, ex: Example) -> str:
        suffix = f" (max {self.max_words} words)" if self.max_words else ""
        return f"{self.instruction}{suffix}\n\nText: {ex.input}\nSummary:"

    def postprocess(self, raw_output: str):
        return raw_output.strip().splitlines()[0]
