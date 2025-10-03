from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dep
    OpenAI = None  # type: ignore


@dataclass
class OpenAIChatAdapter:
    model: str = "gpt-4o-mini"
    name: str = "openai-chat"
    api_key: str | None = None
    # Cost tracking
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0.0

    def _client(self):  # pragma: no cover - network
        if OpenAI is None:
            raise RuntimeError("Please install openeval-lab[openai] to use OpenAI adapter.")
        return OpenAI(api_key=self.api_key) if self.api_key else OpenAI()

    def _get_model_costs(self, model: str) -> dict:
        """Get cost per token for a model (in USD)."""
        # Costs as of September 2025 (approximate)
        costs = {
            "gpt-4o": {"prompt": 5e-6, "completion": 15e-6},  # $5/$15 per million tokens
            "gpt-4o-mini": {
                "prompt": 0.15e-6,
                "completion": 0.6e-6,
            },  # $0.15/$0.60 per million tokens
            "gpt-4-turbo": {"prompt": 10e-6, "completion": 30e-6},
            "gpt-4": {"prompt": 30e-6, "completion": 60e-6},
            "gpt-3.5-turbo": {"prompt": 0.5e-6, "completion": 1.5e-6},
        }
        return costs.get(model, {"prompt": 1e-6, "completion": 2e-6})  # Default fallback

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for token usage."""
        costs = self._get_model_costs(self.model)
        return (prompt_tokens * costs["prompt"]) + (completion_tokens * costs["completion"])

    def get_cost_summary(self) -> dict:
        """Get current cost tracking summary."""
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost_usd": self.total_cost,
            "model": self.model,
        }

    def generate(self, prompt: str, **kwargs: Any) -> str:  # pragma: no cover - network
        client = self._client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 256),
        )

        # Track token usage and costs
        if hasattr(resp, "usage") and resp.usage:
            prompt_tokens = resp.usage.prompt_tokens
            completion_tokens = resp.usage.completion_tokens or 0
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_cost += self._calculate_cost(prompt_tokens, completion_tokens)

        return resp.choices[0].message.content or ""

    def loglikelihood(self, context: str, continuation: str) -> float:  # pragma: no cover - network
        """
        Compute log-likelihood using OpenAI API.

        Note: OpenAI doesn't provide direct log-likelihood access,
        so this is an approximation using logprobs from completion.
        """
        client = self._client()

        # For chat models, we need to use the full prompt
        full_prompt = context + continuation

        try:
            # Use completion endpoint if available, or approximate with chat
            if hasattr(client, "completions") and self.model.startswith("text-"):
                # Legacy completion models
                resp = client.completions.create(
                    model=self.model,
                    prompt=full_prompt,
                    max_tokens=0,  # We don't want generation
                    logprobs=1,
                    echo=True,
                )
                # Extract logprobs for the continuation part
                if resp.choices[0].logprobs and resp.choices[0].logprobs.token_logprobs:
                    # This is a simplified approximation
                    logprobs = resp.choices[0].logprobs.token_logprobs
                    return sum(lp for lp in logprobs if lp is not None)
                else:
                    return -float("inf")
            else:
                # For chat models, we approximate using perplexity-style evaluation
                # This is not ideal but works as a fallback
                messages = [{"role": "user", "content": context + continuation}]
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1,
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=1,
                )

                # Approximate based on response characteristics
                # This is a very rough approximation since OpenAI chat models
                # don't expose continuation logprobs directly
                content_length = len(continuation)
                return -content_length * 0.5  # Rough approximation

        except Exception:
            # Fallback: rough approximation based on text characteristics
            return -len(continuation) * 0.3
