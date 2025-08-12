"""Multiple Choice Question Answering task."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from ..core import Adapter, Example
from ..loglikelihood import MCQATask as BaseMCQATask, LogLikelihoodAdapter
from ..prompt import PromptTemplate


@dataclass
class MCQATask(BaseMCQATask):
    """Multiple Choice Question Answering task using log-likelihood."""

    name: str = "mcqa"

    def __init__(
        self,
        prompt_template: Optional[str] = None,
        normalize_length: bool = True,
    ):
        """Initialize MCQA task."""
        if prompt_template is None:
            # Default template for multiple choice
            prompt_template = """{{question}}
{% for choice in choices %}
{{loop.index0 | alpha}}. {{choice}}
{% endfor %}
Answer:"""

        super().__init__(prompt_template, normalize_length)

    def run(self, adapter: Adapter, ex: Example) -> Dict[str, Any]:
        """Run the task with the adapter."""
        # Check if adapter supports log-likelihood
        if hasattr(adapter, "loglikelihood"):
            # Use log-likelihood evaluation (cast to LogLikelihoodAdapter)
            return self.predict(adapter, ex)  # type: ignore
        else:
            # Fall back to standard text generation
            prompt = self.build_prompt_with_template(ex)
            raw_output = adapter.generate(prompt)
            return {"prediction": self.postprocess(raw_output)}

    def postprocess(self, raw_output: str) -> str:
        """Extract the choice letter from the output."""
        # Look for A, B, C, D pattern in the output
        output = raw_output.strip().upper()

        # Common patterns to extract choice
        if len(output) == 1 and output in "ABCDEFGHIJ":
            return output

        # Look for patterns like "Answer: A" or "A)"
        for letter in "ABCDEFGHIJ":
            if f"ANSWER: {letter}" in output or f"{letter})" in output:
                return letter
            if output.startswith(letter) and (len(output) == 1 or not output[1].isalpha()):
                return letter

        # Default to first letter found
        for char in output:
            if char in "ABCDEFGHIJ":
                return char

        return "A"  # Default fallback

        super().__init__(
            prompt_template=prompt_template,
            normalize_length=normalize_length,
            choice_prefix=" ",  # Space before choice
            choice_suffix="",
        )

    def build_prompt(self, ex: Example) -> str:
        """Build context prompt using template."""
        return self.build_prompt_with_template(ex)

    def get_choices(self, ex: Example) -> List[str]:
        """Extract choices from example."""
        # Try different field names - access via meta or direct attributes
        if hasattr(ex, "choices"):
            choices = getattr(ex, "choices", None)
            if choices:
                return choices

        if ex.meta and "choices" in ex.meta:
            return ex.meta["choices"]

        raise ValueError(f"No choices found in example {ex.id}")

    def get_answer_from_example(self, ex: Example) -> str:
        """Get the correct answer label from example."""
        if hasattr(ex, "answer"):
            answer = getattr(ex, "answer", None)
            if answer:
                return answer

        if ex.reference:
            return str(ex.reference)

        if ex.meta and "answer" in ex.meta:
            return ex.meta["answer"]

        raise ValueError(f"No answer found in example {ex.id}")
