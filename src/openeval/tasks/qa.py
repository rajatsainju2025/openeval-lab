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
        few_shot_examples: Optional[list[dict]] = None,
    ):
        """Initialize QA task with optional template and few-shot examples."""
        self.instruction = instruction
        self.few_shot_examples = few_shot_examples or []

        if prompt_template is None:
            # Default template for QA
            if self.few_shot_examples:
                prompt_template = self._build_few_shot_template()
            else:
                prompt_template = "{{instruction}}\n\nQ: {{input}}\nA:"

        super().__init__(prompt_template)

    def _build_few_shot_template(self) -> str:
        """Build template with few-shot examples."""
        template_parts = [self.instruction, ""]

        for example in self.few_shot_examples:
            template_parts.append(f"Q: {example['input']}")
            template_parts.append(f"A: {example['reference']}")
            template_parts.append("")

        template_parts.append("Q: {{input}}")
        template_parts.append("A:")

        return "\n".join(template_parts)

    def build_prompt(self, ex: Example) -> str:
        """Fallback prompt building if no template is used."""
        if self.few_shot_examples:
            return self._build_few_shot_prompt(ex)
        return f"{self.instruction}\n\nQ: {ex.input}\nA:"

    def _build_few_shot_prompt(self, ex: Example) -> str:
        """Build few-shot prompt."""
        prompt_parts = [self.instruction, ""]

        for example in self.few_shot_examples:
            prompt_parts.append(f"Q: {example['input']}")
            prompt_parts.append(f"A: {example['reference']}")
            prompt_parts.append("")

        prompt_parts.append(f"Q: {ex.input}")
        prompt_parts.append("A:")

        return "\n".join(prompt_parts)

    def build_prompt_with_template(self, ex: Example, **extra_vars) -> str:
        """Build prompt with template, including instruction."""
        extra_vars.setdefault("instruction", self.instruction)
        return super().build_prompt_with_template(ex, **extra_vars)

    def postprocess(self, raw_output: str):
        return raw_output.strip().splitlines()[0]
