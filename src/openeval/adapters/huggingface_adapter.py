"""Hugging Face Transformers adapter for OpenEval."""

import os
from typing import Any, Dict, List, Optional

from ..core import Adapter


class HuggingFaceAdapter(Adapter):
    """Adapter for Hugging Face Transformers models."""

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        device: str = "auto",
        torch_dtype: Optional[str] = None,
        use_auth_token: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
    ):
        """
        Initialize Hugging Face adapter.
        
        Args:
            model_name: HF model name/path
            device: Device to load model on ("auto", "cuda", "cpu")
            torch_dtype: Torch data type ("float16", "bfloat16", etc.)
            use_auth_token: HF auth token for private models
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_auth_token = use_auth_token or os.getenv("HF_TOKEN")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load model and tokenizer."""
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
            except ImportError as e:
                raise ImportError(
                    "transformers and torch required. Install with: "
                    "pip install transformers torch"
                ) from e
            
            # Parse torch dtype
            dtype = None
            if self.torch_dtype == "float16":
                dtype = torch.float16
            elif self.torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_auth_token=self.use_auth_token,
                trust_remote_code=True,
            )
            
            # Set pad token if not available
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "use_auth_token": self.use_auth_token,
            }
            
            if dtype is not None:
                model_kwargs["torch_dtype"] = dtype
                
            if self.device != "auto":
                model_kwargs["device_map"] = self.device
            else:
                model_kwargs["device_map"] = "auto"
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **model_kwargs
            )

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Hugging Face model."""
        self._load_model()
        
        # Override defaults with kwargs
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        do_sample = kwargs.get("do_sample", self.do_sample)
        
        # Tokenize input
        inputs = self._tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        
        # Move to device
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        # Generate
        import torch
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self._tokenizer.decode(
            generated_tokens, 
            skip_special_tokens=True
        )
        
        return generated_text.strip()

    def set_runtime_options(
        self, 
        concurrency: Optional[int] = None, 
        max_retries: Optional[int] = None,
        request_timeout: Optional[float] = None
    ):
        """Set runtime options."""
        # For local models, these don't apply directly
        self._concurrency = concurrency
        self._max_retries = max_retries
        self._request_timeout = request_timeout


class CodeLlamaAdapter(HuggingFaceAdapter):
    """Convenience adapter for Code Llama models."""
    
    def __init__(self, size: str = "7b", **kwargs):
        """
        Initialize Code Llama adapter.
        
        Args:
            size: Model size ("7b", "13b", "34b")
            **kwargs: Additional arguments for HuggingFaceAdapter
        """
        model_name = f"codellama/CodeLlama-{size}-Instruct-hf"
        super().__init__(model_name=model_name, **kwargs)


class LlamaAdapter(HuggingFaceAdapter):
    """Convenience adapter for Llama 2 models."""
    
    def __init__(self, size: str = "7b", chat: bool = True, **kwargs):
        """
        Initialize Llama 2 adapter.
        
        Args:
            size: Model size ("7b", "13b", "70b")
            chat: Whether to use chat variant
            **kwargs: Additional arguments for HuggingFaceAdapter
        """
        variant = "chat" if chat else "hf"
        model_name = f"meta-llama/Llama-2-{size}-{variant}-hf"
        super().__init__(model_name=model_name, **kwargs)
