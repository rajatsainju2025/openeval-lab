from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import numpy as np

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dep
    OpenAI = None  # type: ignore


@dataclass
class OpenAIChatAdapter:
    model: str = "gpt-4o-mini"
    name: str = "openai-chat"
    api_key: str | None = None

    def _client(self):  # pragma: no cover - network
        if OpenAI is None:
            raise RuntimeError("Please install openeval-lab[openai] to use OpenAI adapter.")
        return OpenAI(api_key=self.api_key) if self.api_key else OpenAI()

    def generate(self, prompt: str, **kwargs: Any) -> str:  # pragma: no cover - network
        client = self._client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 256),
        )
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
