from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class EchoAdapter:
    """A trivial adapter that returns the prompt as the output.

    This adapter is primarily used for testing and debugging. It implements
    the full Adapter protocol including async methods and logprobs support.
    """

    name: str = "echo"

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate method that echoes the input prompt."""
        return prompt

    def generate_with_logprobs(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Generate with log probabilities (mock implementation)."""
        return {
            "text": prompt,
            "tokens": prompt.split(),
            "logprobs": [0.0] * len(prompt.split()),
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(prompt.split()),
                "total_tokens": len(prompt.split()) * 2,
            },
        }

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        """Async version of generate for protocol compliance."""
        return self.generate(prompt, **kwargs)

    async def agenerate_with_logprobs(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Async version of generate_with_logprobs for protocol compliance."""
        return self.generate_with_logprobs(prompt, **kwargs)
