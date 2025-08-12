from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from ..core import Example, Task
from ..prompt import PromptTemplate


@dataclass
class QATask(Task):
    name: str = "qa"
    instruction: str = "Answer the question concisely."

    def __init__(
        self,
        instruction: str = "Answer the question concisely.",
        prompt_template: Optional[Union[str, PromptTemplate]] = None,
    ):
        """Initialize QA task with optional template."""
        self.instruction = instruction
        if prompt_template is None:
            # Default template for QA
            prompt_template = "{{instruction}}\n\nQ: {{input}}\nA:"
        super().__init__(prompt_template)

    def build_prompt(self, ex: Example) -> str:
        """Fallback prompt building if no template is used."""
        return f"{self.instruction}\n\nQ: {ex.input}\nA:"

    def build_prompt_with_template(self, ex: Example, **extra_vars) -> str:
        """Build prompt with template, including instruction."""
        extra_vars.setdefault("instruction", self.instruction)
        return super().build_prompt_with_template(ex, **extra_vars)

    def postprocess(self, raw_output: str):
        return raw_output.strip().splitlines()[0]
