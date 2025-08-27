from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class EchoAdapter:
    """A trivial adapter that returns the prompt as the output."""

    name: str = "echo"

    def generate(self, prompt: str, **kwargs) -> str:
        return prompt

    def generate_with_logprobs(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Optional capability for protocol compliance."""
        return {
            "text": prompt,
            "tokens": prompt.split(),
            "logprobs": [0.0] * len(prompt.split()),
            "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(prompt.split()), "total_tokens": len(prompt.split()) * 2},
        }
