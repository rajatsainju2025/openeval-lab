from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EchoAdapter:
    """A trivial adapter that returns the prompt as the output."""

    name: str = "echo"

    def generate(self, prompt: str, **kwargs) -> str:
        return prompt
