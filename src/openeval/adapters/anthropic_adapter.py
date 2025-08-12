"""Anthropic Claude adapter for OpenEval."""

import os
from typing import Any, Dict, Optional

from ..core import Adapter


class AnthropicAdapter(Adapter):
    """Adapter for Anthropic Claude models."""

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        """
        Initialize Anthropic adapter.
        
        Args:
            model: Model name (e.g., claude-3-haiku-20240307, claude-3-sonnet-20240229)
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

    def _client(self):
        """Get Anthropic client."""
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            ) from e
        
        return anthropic.Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic Claude."""
        client = self._client()
        
        # Override defaults with kwargs
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            # Graceful degradation for network issues
            raise RuntimeError(f"Anthropic API error: {e}") from e

    def set_runtime_options(
        self, 
        concurrency: Optional[int] = None, 
        max_retries: Optional[int] = None,
        request_timeout: Optional[float] = None
    ):
        """Set runtime options (placeholder for future implementation)."""
        # Note: Anthropic client doesn't directly support these options
        # but we can store them for future use
        self._concurrency = concurrency
        self._max_retries = max_retries  
        self._request_timeout = request_timeout


class ClaudeHaikuAdapter(AnthropicAdapter):
    """Convenience adapter for Claude 3 Haiku."""
    
    def __init__(self, **kwargs):
        super().__init__(model="claude-3-haiku-20240307", **kwargs)


class ClaudeSonnetAdapter(AnthropicAdapter):
    """Convenience adapter for Claude 3 Sonnet."""
    
    def __init__(self, **kwargs):
        super().__init__(model="claude-3-sonnet-20240229", **kwargs)


class ClaudeOpusAdapter(AnthropicAdapter):
    """Convenience adapter for Claude 3 Opus."""
    
    def __init__(self, **kwargs):
        super().__init__(model="claude-3-opus-20240229", **kwargs)
