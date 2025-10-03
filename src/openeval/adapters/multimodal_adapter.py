from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import base64
from PIL import Image
import io

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dep
    OpenAI = None  # type: ignore


@dataclass
class OpenAIMultimodalAdapter:
    """OpenAI Vision API adapter for multimodal evaluation."""
    model: str = "gpt-4o-mini"
    name: str = "openai-multimodal"
    api_key: str | None = None
    supported_modalities: Optional[List[str]] = None
    # Cost tracking
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0.0

    def __post_init__(self):
        if self.supported_modalities is None:
            self.supported_modalities = ["text", "image"]

    def _client(self):  # pragma: no cover - network
        if OpenAI is None:
            raise RuntimeError("Please install openeval-lab[openai] to use OpenAI multimodal adapter.")
        return OpenAI(api_key=self.api_key) if self.api_key else OpenAI()

    def encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64 for API transmission."""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def decode_image(self, base64_string: str) -> Image.Image:
        """Decode base64 string to PIL Image."""
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))

    def validate_multimodal_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate that input contains supported modalities."""
        modalities = input_data.get('modalities', ['text'])
        return all(mod in self.supported_modalities for mod in modalities)

    def predict_multimodal(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multimodal inputs using OpenAI Vision API."""
        results = []

        for input_data in inputs:
            try:
                messages = self._build_multimodal_messages(input_data)

                response = self._client().chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.0
                )

                prediction = response.choices[0].message.content

                # Update cost tracking
                if response.usage:
                    self.total_prompt_tokens += response.usage.prompt_tokens
                    self.total_completion_tokens += response.usage.completion_tokens
                    self.total_cost += self._calculate_cost(
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens
                    )

                results.append({
                    "prediction": prediction,
                    "usage": response.usage.model_dump() if response.usage else {},
                    "model": response.model
                })

            except Exception as e:
                results.append({
                    "prediction": "",
                    "error": str(e),
                    "model": self.model
                })

        return results

    def _build_multimodal_messages(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build messages for multimodal input."""
        messages = []

        # System message if provided
        if "system" in input_data:
            messages.append({"role": "system", "content": input_data["system"]})

        # User message with multimodal content
        user_content = []

        # Add text content
        if "input" in input_data:
            user_content.append({"type": "text", "text": input_data["input"]})

        # Add image content
        if "images" in input_data:
            for image_data in input_data["images"]:
                if isinstance(image_data, str) and image_data.startswith("data:"):
                    # Base64 encoded image
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": image_data}
                    })
                elif isinstance(image_data, str):
                    # File path
                    base64_image = self.encode_image(image_data)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    })

        messages.append({"role": "user", "content": user_content})
        return messages

    def _get_model_costs(self, model: str) -> dict:
        """Get cost per token for a model (in USD)."""
        # Costs as of September 2025 (approximate)
        costs = {
            "gpt-4o": {"prompt": 5e-6, "completion": 15e-6},  # $5/$15 per million tokens
            "gpt-4o-mini": {"prompt": 0.15e-6, "completion": 0.6e-6},  # $0.15/$0.60 per million tokens
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
            "total_cost": self.total_cost,
            "supported_modalities": self.supported_modalities
        }
