"""Local API adapter for models served via HTTP API (e.g., Ollama, local servers)."""

import json

import httpx

from ..enhanced_logging import get_logger

logger = get_logger(__name__)


class LocalAPIAdapter:
    """Adapter for local models served via HTTP API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama2",
        temperature: float = 0.7,
        max_tokens: int = 100,
        **kwargs
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = f"local-{model}"
        self.client = httpx.Client(timeout=30.0)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from local API."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "stream": False
            }

            response = self.client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "").strip()

        except httpx.RequestError as e:
            logger.error(f"Request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()
