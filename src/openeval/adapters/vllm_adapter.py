"""vLLM adapter for high-throughput inference."""

from typing import Any, Dict, List, Optional
import os

from ..core import Adapter


class VLLMAdapter(Adapter):
    """Adapter for vLLM models for high-throughput inference."""

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = -1,
    ):
        """
        Initialize vLLM adapter.
        
        Args:
            model_name: Model name/path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization
            max_model_len: Maximum model length
            dtype: Data type ("auto", "float16", "bfloat16")
            trust_remote_code: Whether to trust remote code
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
        """
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
        self._llm = None
        self._sampling_params = None

    def _load_model(self):
        """Lazy load vLLM model."""
        if self._llm is None:
            try:
                from vllm import LLM, SamplingParams
            except ImportError as e:
                raise ImportError(
                    "vllm required. Install with: pip install vllm"
                ) from e
            
            # Model arguments
            model_kwargs = {
                "model": self.model_name,
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "trust_remote_code": self.trust_remote_code,
                "dtype": self.dtype,
            }
            
            if self.max_model_len is not None:
                model_kwargs["max_model_len"] = self.max_model_len
            
            self._llm = LLM(**model_kwargs)
            
            # Sampling parameters
            self._sampling_params = SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k if self.top_k > 0 else None,
            )

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using vLLM."""
        self._load_model()
        
        # Override sampling parameters if provided
        sampling_params = self._sampling_params
        if any(k in kwargs for k in ["max_new_tokens", "temperature", "top_p", "top_k"]):
            sampling_params = SamplingParams(
                max_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                top_k=kwargs.get("top_k", self.top_k) if kwargs.get("top_k", self.top_k) > 0 else None,
            )
        
        # Generate
        outputs = self._llm.generate([prompt], sampling_params)
        
        # Extract generated text
        generated_text = outputs[0].outputs[0].text
        
        return generated_text.strip()

    def set_runtime_options(
        self, 
        concurrency: Optional[int] = None, 
        max_retries: Optional[int] = None,
        request_timeout: Optional[float] = None
    ):
        """Set runtime options."""
        self._concurrency = concurrency
        self._max_retries = max_retries
        self._request_timeout = request_timeout


class VLLMCodeLlamaAdapter(VLLMAdapter):
    """Convenience adapter for Code Llama models with vLLM."""
    
    def __init__(self, size: str = "7b", **kwargs):
        """
        Initialize Code Llama vLLM adapter.
        
        Args:
            size: Model size ("7b", "13b", "34b")
            **kwargs: Additional arguments for VLLMAdapter
        """
        model_name = f"codellama/CodeLlama-{size}-Instruct-hf"
        super().__init__(model_name=model_name, **kwargs)


class VLLMLlamaAdapter(VLLMAdapter):
    """Convenience adapter for Llama models with vLLM."""
    
    def __init__(self, size: str = "7b", **kwargs):
        """
        Initialize Llama vLLM adapter.
        
        Args:
            size: Model size ("7b", "13b", "70b")
            **kwargs: Additional arguments for VLLMAdapter
        """
        model_name = f"meta-llama/Llama-2-{size}-hf"
        super().__init__(model_name=model_name, **kwargs)
