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
