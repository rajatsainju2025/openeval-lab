"""Adapter protocol for model API interfaces."""

from __future__ import annotations

from typing import Any, Dict, Protocol


class Adapter(Protocol):
    """Protocol defining the interface for model API adapters.

    An Adapter provides a standardized interface for interacting with different language
    models and APIs (e.g., OpenAI, Hugging Face, etc.). It handles the details of making
    requests to the model and processing responses.

    At minimum, adapters must implement the synchronous `generate` method. They may
    optionally implement async versions and/or methods that return additional information
    like token probabilities.

    Attributes:
        name: A unique identifier for the adapter.
    """

    name: str

    def generate(self, prompt: str, **kwargs: Any) -> str:  # sync for simplicity first
        """Generate a completion for the given prompt.

        Args:
            prompt: The input prompt string to send to the model.
            **kwargs: Additional model-specific arguments (e.g., temperature, max_tokens).

        Returns:
            The model's generated text response.
        """
        ...

    def generate_with_logprobs(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Optional method to get generation with token probabilities and usage stats.

        Args:
            prompt: The input prompt string to send to the model.
            **kwargs: Additional model-specific arguments.

        Returns:
            A dictionary containing:
                text: The generated text response
                tokens: List of tokens in the response
                logprobs: List of log probabilities for each token
                usage: Token counts for prompt and completion

        Raises:
            NotImplementedError: If the adapter doesn't support this capability.
        """
        raise NotImplementedError

    # Optional async methods for improved throughput
    async def agenerate(self, prompt: str, **kwargs: Any) -> str:  # pragma: no cover - optional
        """Async version of generate. Fallback to sync if not implemented."""
        return self.generate(prompt, **kwargs)

    async def agenerate_with_logprobs(
        self, prompt: str, **kwargs: Any
    ) -> Dict[str, Any]:  # pragma: no cover - optional
        """Async version of generate_with_logprobs."""
        return self.generate_with_logprobs(prompt, **kwargs)


__all__ = ["Adapter"]
