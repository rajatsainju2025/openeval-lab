"""Core module containing the fundamental abstractions and interfaces.

This module defines the core protocols and abstract base classes that form the
foundation of the evaluation framework:

1. Example - A single evaluation instance with input and reference
2. Dataset - Collection of examples for evaluation
3. Task - Defines how to evaluate model on specific capability
4. Adapter - Interface to model API or local model
5. Metric - Computes evaluation scores

Together these components enable flexible and reproducible model evaluation:

- Tasks define how to format inputs and handle outputs
- Datasets provide evaluation examples with references
- Adapters abstract away model-specific details
- Metrics compute quantitative performance scores

Example:
    >>> from openeval.tasks import QATask
    >>> from openeval.datasets import JSONLDataset 
    >>> from openeval.adapters import OpenAIAdapter
    >>> from openeval.metrics import ExactMatch
    
    >>> task = QATask()
    >>> dataset = JSONLDataset("examples.jsonl")
    >>> adapter = OpenAIAdapter(model="gpt-4")
    >>> metric = ExactMatch()
    
    >>> results = task.evaluate(adapter, dataset, [metric])
    >>> print(f"Accuracy: {results['metrics']['exact_match']['accuracy']:.2%}")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Protocol, Union
from pathlib import Path
import time
import sys
import platform
from importlib.metadata import version as _pkg_version, PackageNotFoundError
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import set_seed, hash_file, retry_call, run_with_timeout, hash_prompt
from .cache import PredictionCache, CacheStats
from .prompt import PromptTemplate
from .enhanced_logging import get_logger

logger = get_logger(__name__)


def _categorize_error(err: Exception) -> str:
    """Categorize an exception into standardized error types for consistent reporting.
    
    Args:
        err: The exception to categorize.
        
    Returns:
        A standardized error category string, one of:
        - TIMEOUT: Operation timed out
        - RATE_LIMIT: Rate limit exceeded (HTTP 429)
        - NETWORK: Network/connection issues
        - AUTH: Authentication failures (HTTP 401/403)
        - QUOTA: Resource quota exceeded (HTTP 402)
        - SERVER_ERROR: Server-side errors (HTTP 500/502/503)
        - INVALID_REQUEST: Invalid request (HTTP 400)
        - {Exception.__name__}: Other exceptions, using exception type name
    """
    err_str = str(err).lower()
    err_type = type(err).__name__
    
    if "timeout" in err_str or "timed out" in err_str or isinstance(err, TimeoutError):
        return "TIMEOUT"
    elif "rate limit" in err_str or "429" in err_str:
        return "RATE_LIMIT"
    elif "connection" in err_str or "network" in err_str:
        return "NETWORK"
    elif "authentication" in err_str or "401" in err_str or "403" in err_str:
        return "AUTH"
    elif "quota" in err_str or "402" in err_str:
        return "QUOTA"
    elif "server" in err_str or "500" in err_str or "502" in err_str or "503" in err_str:
        return "SERVER_ERROR"
    elif "invalid" in err_str or "400" in err_str:
        return "INVALID_REQUEST"
    else:
        return f"{err_type}"


def _summarize_errors(per_error: List[Optional[str]]) -> Dict[str, int]:
    """Count and summarize errors by their category.
    
    Takes a list of error messages (potentially including Nones) and produces a count
    by error category. Error categories are expected to be in the format [CATEGORY]message.
    
    Args:
        per_error: List of error messages, where each message may be None or a string.
                  Strings starting with [CATEGORY] will be counted under that category.
    
    Returns:
        A dictionary mapping error categories to their counts. Unknown categories are
        counted under the "UNKNOWN" key.
    
    Example:
        >>> _summarize_errors(["[TIMEOUT]Request timed out", None, "[TIMEOUT]Another timeout"])
        {'TIMEOUT': 2}
    """
    error_counts: Dict[str, int] = {}
    for error in per_error:
        if error:
            # Extract category from [CATEGORY] message format
            if error.startswith("[") and "]" in error:
                category = error.split("]")[0][1:]
            else:
                category = "UNKNOWN"
            error_counts[category] = error_counts.get(category, 0) + 1
    return error_counts


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

    async def agenerate_with_logprobs(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - optional
        """Async version of generate_with_logprobs."""
        return self.generate_with_logprobs(prompt, **kwargs)


class Metric(Protocol):
    """Protocol defining evaluation metrics for model outputs.
    
    A Metric computes quantitative scores comparing model predictions against reference
    answers. Metrics can range from simple exact match to complex semantic similarity
    measures. They must be deterministic and return consistent scores for the same
    inputs.
    
    Common metric types include:
    - Accuracy metrics (exact match, case-insensitive match)
    - Partial match metrics (F1 score, ROUGE)
    - Semantic metrics (BERTScore, embedding similarity)
    - Task-specific metrics (BLEU for translation, perplexity for LMs)
    
    Invariants:
        - Must be deterministic given same inputs
        - Must handle batched inputs efficiently
        - Should be robust to common input variations
        - Should validate inputs and raise informative errors
    
    Attributes:
        name: A unique identifier for this metric.
    """

    name: str

    def compute(
        self, predictions: Iterable[Any], references: Iterable[Any]
    ) -> Mapping[str, float]:
        """Compute evaluation scores comparing predictions to references.
        
        Both inputs must be iterables of the same length. The metric may compute
        multiple related scores (e.g., precision/recall/F1) and return them in
        a dictionary.
        
        Args:
            predictions: Model outputs to evaluate.
            references: Expected correct outputs to compare against.
            
        Returns:
            Dictionary mapping score names to float values.
            Common keys include:
            - accuracy: Fraction of exact matches
            - f1: F1 score for partial matches
            - rouge_1/2/L: ROUGE scores for summarization
            - bleu: BLEU score for translation
            
        Raises:
            ValueError: If inputs are invalid or incompatible.
        """
        ...


@dataclass
class Example:
    """A single evaluation example containing input, reference answer, and metadata.
    
    An Example represents one instance in an evaluation dataset. It contains the input
    that will be given to the model (after task-specific prompt construction), the
    reference answer(s) that will be used to evaluate the model's output, and optional
    metadata about the example.
    
    Attributes:
        id: A unique identifier for the example within its dataset.
        input: The raw input that will be processed by the task's prompt template.
            Can be a string for simple QA tasks or structured data for complex tasks.
        reference: The expected output or "ground truth" answer. Can be a string,
            list of strings for multiple references, or structured data.
        meta: Optional dictionary of metadata about this example (e.g., difficulty,
            source, tags). Accessible in prompt templates.
    
    Example:
        >>> example = Example(
        ...     id="qa-1",
        ...     input="What is the capital of France?",
        ...     reference="Paris",
        ...     meta={"difficulty": "easy", "category": "geography"}
        ... )
    """
    id: str
    input: Any
    reference: Any
    meta: Optional[Dict[str, Any]] = None


class Dataset(ABC):
    """Abstract base class for evaluation datasets.
    
    A Dataset provides an iterable interface over evaluation Examples. It represents
    a collection of inputs and their corresponding reference outputs/answers that
    will be used to evaluate model performance.
    
    Implementations must provide an iterator over Examples. They should also provide
    a meaningful name that identifies the dataset. The length (number of examples)
    is computed automatically if not overridden.
    
    Invariants:
        - Iterator must be deterministic given a seed
        - Examples must have unique IDs within the dataset
        - Must support multiple iterations (reusable)
    
    Attributes:
        name: A unique identifier for this dataset implementation.
    
    Example:
        >>> class QADataset(Dataset):
        ...     name = "qa_dataset"
        ...     def __iter__(self):
        ...         yield Example(id="1", input="What is 2+2?", reference="4")
        ...         yield Example(id="2", input="What is pi?", reference="3.14159")
    """
    name: str

    @abstractmethod
    def __iter__(self) -> Iterator[Example]:
        """Iterate over examples in the dataset.
        
        Returns:
            An iterator over Example instances.
        
        Note:
            - Must be reusable (support multiple iterations)
            - Must be deterministic if seed is set
            - Should lazy load if possible for large datasets
        """
        ...

    def __len__(self) -> int:
        """Get the number of examples in the dataset.
        
        Returns:
            The total number of examples.
            
        Note:
            Default implementation consumes the iterator.
            Override for more efficient implementation.
        """
        return sum(1 for _ in iter(self))


class Task(ABC):
    """Abstract base class for evaluation tasks.
    
    A Task defines how to evaluate a model on a particular capability or behavior.
    It handles converting dataset examples into model-appropriate prompts and
    post-processing model outputs for evaluation.
    
    Tasks can use either a custom prompt-building implementation or a template-based
    approach. The template approach is recommended for simpler tasks as it provides
    better reproducibility and easier modification.
    
    Invariants:
        - Prompt building must be deterministic for given example and seed
        - Prompts must be valid for the target model/adapter
        - Post-processing must be consistent and preserve evaluation-critical information
    
    Attributes:
        name: A unique identifier for this task implementation.
        prompt_template: Optional template for generating prompts from examples.
    """
    name: str

    def __init__(self, prompt_template: Optional[Union[str, PromptTemplate]] = None):
        """Initialize task with optional prompt template.
        
        Args:
            prompt_template: Either a string template or PromptTemplate instance.
                If a string is provided, it will be converted to a PromptTemplate.
        """
        self._prompt_template_raw = prompt_template
        if isinstance(prompt_template, str):
            self.prompt_template = PromptTemplate(prompt_template)
        else:
            self.prompt_template = prompt_template

    @abstractmethod
    def build_prompt(self, ex: Example) -> str:
        """Convert an example into a model-ready prompt string.
        
        This is the core method that defines how examples are presented to the model.
        Must be implemented by concrete task classes unless using templates.
        
        Args:
            ex: The example to convert into a prompt.
            
        Returns:
            A string prompt ready to be sent to the model.
        """
        ...

    def build_prompt_with_template(self, ex: Example, **extra_vars: Any) -> str:
        """Build prompt using template if available, otherwise fallback to build_prompt.
        
        This method is used when a prompt template is provided. The template can
        access all example fields (input, reference, id) plus any metadata fields
        and extra variables provided.
        
        Args:
            ex: The example to convert into a prompt.
            **extra_vars: Additional variables to make available to the template.
            
        Returns:
            The rendered prompt string.
        """
        if self.prompt_template is not None:
            # Prepare template variables
            variables = {"input": ex.input, "reference": ex.reference, "id": ex.id, **extra_vars}
            # Add meta fields as top-level variables
            if ex.meta:
                variables.update(ex.meta)

            return self.prompt_template.render(**variables)
        else:
            return self.build_prompt(ex)

    def postprocess(self, raw_output: str) -> Any:
        """Post-process raw model output into evaluation-ready format.
        
        This method can be overridden to implement custom output processing like:
        - Extracting specific answer formats
        - Normalizing whitespace or case
        - Converting to structured data
        
        Args:
            raw_output: The raw string output from the model.
            
        Returns:
            Processed output ready for metric computation.
        """
        return raw_output.strip()

    def evaluate(
        self,
        adapter: Adapter,
        dataset: Dataset,
        metrics: List[Metric],
        *,
        seed: Optional[int] = 0,
        collect_records: bool = False,
        concurrency: int = 1,
        max_retries: int = 0,
        request_timeout: Optional[float] = None,
        streaming_batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate with optional streaming for memory efficiency.
        
        Args:
            streaming_batch_size: If provided, process examples in batches of this size
                                to reduce memory usage for large datasets.
        """
        set_seed(seed)
        
        # Use streaming evaluation for large datasets if batch_size specified
        if streaming_batch_size and streaming_batch_size > 0:
            return self._evaluate_streaming(
                adapter, dataset, metrics, seed, collect_records, 
                concurrency, max_retries, request_timeout, streaming_batch_size
            )
        
        # Original evaluation logic for smaller datasets
        examples: List[Example] = list(iter(dataset))
        n = len(examples)
        predictions: List[Any] = [None] * n
        references: List[Any] = [None] * n
        per_latency: List[float] = [0.0] * n
        per_error: List[Optional[str]] = [None] * n
        per_cached: List[bool] = [False] * n

        success_count = 0
        error_count = 0

        # Cache plumbing (adapter attributes set by CLI)
        cache_mode = str(getattr(adapter, "_cache_mode", "off")).lower()
        cache_dir = getattr(adapter, "_cache_dir", None)
        cache_ttl = getattr(adapter, "_cache_ttl", None)
        cache: Optional[PredictionCache] = None
        cache_stats = CacheStats()
        if cache_mode != "off" and cache_dir is not None:
            try:
                cache = PredictionCache(Path(cache_dir))
            except Exception:
                cache = None

        def _cache_key(prompt: str) -> str:
            adapter_name = getattr(adapter, "name", adapter.__class__.__name__)
            model = getattr(adapter, "model", None)
            temp = getattr(adapter, "temperature", None)
            system = getattr(adapter, "system_prompt", None)
            key_mode = str(getattr(adapter, "_cache_key_mode", "strict")).lower()
            parts: List[Any] = [adapter_name, prompt]
            if key_mode == "strict":
                parts.extend([model, temp, system])
            return hash_prompt(parts)

        def _maybe_read_cache(prompt: str) -> Optional[str]:
            if cache is None or cache_mode not in {"read", "rw", "write"}:
                return None
            if cache_mode == "write":
                return None
            try:
                val = cache.get(_cache_key(prompt), ttl=cache_ttl)
            except Exception:
                return None
            if val is not None:
                cache_stats.hits += 1
            else:
                cache_stats.misses += 1
            return val

        def _maybe_write_cache(prompt: str, output: str) -> None:
            if cache is None or cache_mode not in {"write", "rw"}:
                return
            try:
                cache.set(_cache_key(prompt), output)
            except Exception:
                pass

        def _call_generate(prompt: str) -> str:
            try:
                cached = _maybe_read_cache(prompt)
                if cached is not None:
                    return cached
                out = retry_call(
                    lambda: run_with_timeout(lambda: adapter.generate(prompt), request_timeout),
                    retries=max_retries,
                )
                _maybe_write_cache(prompt, out)
                return out
            except Exception as e:
                logger.error(f"Failed to generate response for prompt: {e}", exc_info=True)
                raise

        t0 = time.perf_counter()
        
        # Performance optimization: Use memory-efficient processing for large datasets
        def _get_memory_usage():
            """Get current memory usage in MB."""
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                return process.memory_info().rss / 1024 / 1024
            except ImportError:
                return 0
        
        initial_memory = _get_memory_usage()
        peak_memory = initial_memory
        
        # Optimize concurrency based on available resources
        try:
            import psutil
            import os
            if concurrency > 1:
                # Adjust concurrency based on system resources
                cpu_count = os.cpu_count() or 4
                available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024  # GB
                
                # Conservative concurrency scaling
                optimal_concurrency = min(concurrency, cpu_count * 2, max(1, int(available_memory / 2)))
                if optimal_concurrency != concurrency:
                    print(f"Adjusting concurrency from {concurrency} to {optimal_concurrency} based on system resources")
                    concurrency = optimal_concurrency
        except ImportError:
            pass

        # Validate dataset before processing
        try:
            examples = list(iter(dataset))
            if not examples:
                raise ValueError("Dataset is empty")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}", exc_info=True)
            raise ValueError(f"Dataset loading failed: {e}") from e
        
        if max(1, int(concurrency)) <= 1:
            for i, ex in enumerate(examples):
                references[i] = ex.reference
                prompt = self.build_prompt_with_template(ex)
                s = time.perf_counter()
                try:
                    cached = _maybe_read_cache(prompt)
                    if cached is not None:
                        raw = cached
                        per_cached[i] = True
                    else:
                        raw = retry_call(
                            lambda: run_with_timeout(
                                lambda: adapter.generate(prompt), request_timeout
                            ),
                            retries=max_retries,
                        )
                        _maybe_write_cache(prompt, raw)
                    e = time.perf_counter()
                    per_latency[i] = e - s
                    success_count += 1
                    predictions[i] = self.postprocess(raw)
                except Exception as err:  # pragma: no cover - depends on adapter
                    e = time.perf_counter()
                    per_latency[i] = e - s
                    error_count += 1
                    
                    # Categorize error for better diagnostics
                    error_category = _categorize_error(err)
                    detailed_error = f"[{error_category}] {str(err)}"
                    per_error[i] = detailed_error
                    predictions[i] = ""
        else:
            with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:  # pragma: no cover
                futures = []
                for i, ex in enumerate(examples):
                    references[i] = ex.reference
                    prompt = self.build_prompt_with_template(ex)

                    def make_job(idx: int, pr: str):
                        def _job():
                            s = time.perf_counter()
                            try:
                                cached = _maybe_read_cache(pr)
                                if cached is not None:
                                    raw = cached
                                    cached_flag = True
                                else:
                                    raw = retry_call(
                                        lambda: run_with_timeout(
                                            lambda: adapter.generate(pr), request_timeout
                                        ),
                                        retries=max_retries,
                                    )
                                    _maybe_write_cache(pr, raw)
                                    cached_flag = False
                                e = time.perf_counter()
                                return idx, self.postprocess(raw), (e - s), None, cached_flag
                            except Exception as err:
                                e = time.perf_counter()
                                error_category = _categorize_error(err)
                                detailed_error = f"[{error_category}] {str(err)}"
                                return idx, "", (e - s), detailed_error, False

                        return _job

                    futures.append(pool.submit(make_job(i, prompt)))
                for fut in as_completed(futures):
                    idx, pred, dur, err, cached_flag = fut.result()
                    predictions[idx] = pred
                    per_latency[idx] = dur
                    per_cached[idx] = cached_flag
                    if err is None:
                        success_count += 1
                    else:
                        error_count += 1


        total_duration = time.perf_counter() - t0
        latencies = [x for x in per_latency if x > 0]
        
        # Update peak memory usage
        try:
            current_memory = _get_memory_usage()
            peak_memory = max(peak_memory, current_memory)
        except NameError:
            current_memory = 0
        
        results: Dict[str, Any] = {}
        for m in metrics:
            try:
                results[m.name] = m.compute(predictions, references)
            except Exception as err:
                # Record the error string so UIs can show unavailable metrics
                results[m.name] = {"error": f"{err}"}

        import datetime as _dt

        # Build manifest for reproducibility
        def _maybe_ver(pkg: str) -> Optional[str]:
            try:
                return _pkg_version(pkg)
            except PackageNotFoundError:
                return None
            except Exception:
                return None

        # Try to include git commit hash if available
        git: Dict[str, Any] = {}
        try:
            import subprocess

            # Use git rev-parse in the project root; if it fails, ignore
            rev = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )
            git["commit"] = rev
        except Exception:
            pass

        # CUDA and environment snapshot (best effort)
        try:
            import subprocess as _sp
            has_nvidia_smi = (
                _sp.run(["nvidia-smi"], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL).returncode == 0
            )
        except Exception:
            has_nvidia_smi = False
        import os as _os
        _cuda_info = {
            "cuda_visible_devices": _os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_smi": has_nvidia_smi,
        }
        _env_info = {k: _os.environ.get(k) for k in ["LANG", "LC_ALL", "TZ"]}

        manifest: Dict[str, Any] = {
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "openeval_version": _maybe_ver("openeval-lab"),
            "python": {
                "version": sys.version.split()[0],
                "executable": sys.executable,
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "packages": {
                k: v
                for k, v in {
                    "fastapi": _maybe_ver("fastapi"),
                    "jinja2": _maybe_ver("jinja2"),
                    "numpy": _maybe_ver("numpy"),
                    "pandas": _maybe_ver("pandas"),
                    "sacrebleu": _maybe_ver("sacrebleu"),
                    "bert-score": _maybe_ver("bert-score"),
                    "openai": _maybe_ver("openai"),
                    "datasets": _maybe_ver("datasets"),
                }.items()
                if v is not None
            },
            "git": git if git else None,
            "cuda": _cuda_info,
            "env": _env_info,
            "adapter": {
                "name": getattr(adapter, "name", adapter.__class__.__name__),
                "class": f"{adapter.__class__.__module__}.{adapter.__class__.__name__}",
            },
            "task": {
                "name": getattr(self, "name", self.__class__.__name__),
                "class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            },
        }
        
        # Add cost information if available
        cost_method = getattr(adapter, 'get_cost_summary', None)
        if cost_method and callable(cost_method):
            try:
                cost_info = cost_method()
                manifest["cost"] = cost_info
            except Exception:
                pass
        
        # Drop None git if not available
        if manifest.get("git") is None:
            manifest.pop("git", None)

        payload: Dict[str, Any] = {
            "task": self.name,
            "dataset": getattr(dataset, "name", dataset.__class__.__name__),
            "size": len([p for p in predictions if p is not None]),
            "metrics": results,
            "adapter": getattr(adapter, "name", adapter.__class__.__name__),
            "seed": seed,
            "timing": {
                "avg_latency_ms": (sum(latencies) / len(latencies) * 1000.0) if latencies else 0.0,
                "total_seconds": total_duration,
                "throughput_eps": ((n / total_duration) if total_duration > 0 else 0.0),
                "request_successes": success_count,
                "request_errors": error_count,
                "error_rate": (error_count / n) if n > 0 else 0.0,
                "cache_hits": cache_stats.hits,
                "cache_misses": cache_stats.misses,
                "cache_hit_rate": cache_stats.hit_rate,
                "memory_usage_mb": {
                    "initial": initial_memory,
                    "peak": peak_memory,
                    "current": _get_memory_usage()
                } if initial_memory > 0 else None,
            },
            
            # Add error categorization summary
            "error_summary": _summarize_errors(per_error),
            "manifest": manifest,
        }
        # Attempt to add pip freeze for reproducibility (best effort)
        try:
            import subprocess as _sp
            freeze = _sp.check_output([sys.executable, "-m", "pip", "freeze"], stderr=_sp.DEVNULL).decode().splitlines()
            payload.setdefault("environment", {})["pip_freeze"] = freeze
        except Exception:
            pass
        # dataset fingerprint if file-backed
        ds_path = getattr(dataset, "path", None)
        if ds_path is not None:
            p = Path(ds_path)
            if p.is_file():
                try:
                    payload["dataset_path"] = str(p)
                    payload["dataset_hash_sha256"] = hash_file(p)
                except Exception:
                    pass
        if collect_records:
            records: List[Dict[str, Any]] = []
            for i, ex in enumerate(examples):
                rec: Dict[str, Any] = {
                    "id": ex.id,
                    "input": ex.input,
                    "reference": ex.reference,
                    "prediction": predictions[i],
                    "latency_ms": per_latency[i] * 1000.0,
                }
                if per_error[i] is not None:
                    rec["error"] = per_error[i]
                if per_cached[i]:
                    rec["cached"] = True
                records.append(rec)
            payload["records"] = records
        # close cache connection
        if cache is not None:
            try:
                cache.close()
            except Exception:
                pass
        return payload

    def _evaluate_streaming(
        self,
        adapter: Adapter,
        dataset: Dataset,
        metrics: List[Metric],
        seed: Optional[int],
        collect_records: bool,
        concurrency: int,
        max_retries: int,
        request_timeout: Optional[float],
        batch_size: int,
    ) -> Dict[str, Any]:
        """
        Memory-efficient streaming evaluation for large datasets.
        Processes examples in batches to minimize memory usage.
        """
        set_seed(seed)
        
        # Initialize tracking variables
        all_predictions: List[Any] = []
        all_references: List[Any] = []
        all_latencies: List[float] = []
        all_errors: List[Optional[str]] = []
        all_cached: List[bool] = []
        
        success_count = 0
        error_count = 0
        total_examples = 0
        
        # Cache setup
        cache_mode = str(getattr(adapter, "_cache_mode", "off")).lower()
        cache_dir = getattr(adapter, "_cache_dir", None)
        cache_ttl = getattr(adapter, "_cache_ttl", None)
        cache: Optional[PredictionCache] = None
        cache_stats = CacheStats()
        if cache_mode != "off" and cache_dir is not None:
            try:
                cache = PredictionCache(Path(cache_dir))
            except Exception:
                cache = None

        # Cache helper functions
        def _cache_key(prompt: str) -> str:
            adapter_name = getattr(adapter, "name", adapter.__class__.__name__)
            model = getattr(adapter, "model", None)
            temp = getattr(adapter, "temperature", None)
            system = getattr(adapter, "system_prompt", None)
            key_mode = str(getattr(adapter, "_cache_key_mode", "strict")).lower()
            parts: List[Any] = [adapter_name, prompt]
            if key_mode == "strict":
                parts.extend([model, temp, system])
            return hash_prompt(parts)

        def _maybe_read_cache(prompt: str) -> Optional[str]:
            if cache is None or cache_mode not in {"read", "rw", "write"}:
                return None
            if cache_mode == "write":
                return None
            try:
                val = cache.get(_cache_key(prompt), ttl=cache_ttl)
                if val is not None:
                    cache_stats.hits += 1
                else:
                    cache_stats.misses += 1
                return val
            except Exception:
                return None

        def _maybe_write_cache(prompt: str, output: str) -> None:
            if cache is None or cache_mode not in {"write", "rw"}:
                return
            try:
                cache.set(_cache_key(prompt), output)
            except Exception:
                pass

        def _call_generate(prompt: str) -> str:
            try:
                cached = _maybe_read_cache(prompt)
                if cached is not None:
                    return cached
                out = retry_call(
                    lambda: run_with_timeout(lambda: adapter.generate(prompt), request_timeout),
                    retries=max_retries,
                )
                _maybe_write_cache(prompt, out)
                return out
            except Exception as e:
                logger.error(f"Failed to generate response for prompt: {e}", exc_info=True)
                raise

        # Process dataset in batches
        batch_examples = []
        batch_indices = []
        
        t0 = time.perf_counter()
        
        for i, ex in enumerate(iter(dataset)):
            batch_examples.append(ex)
            batch_indices.append(i)
            total_examples += 1
            
            # Process batch when it reaches the specified size
            if len(batch_examples) >= batch_size:
                self._process_batch(
                    batch_examples, batch_indices, adapter, all_predictions, 
                    all_references, all_latencies, all_errors, all_cached,
                    success_count, error_count, _call_generate, _maybe_read_cache, 
                    _maybe_write_cache, concurrency, max_retries, request_timeout
                )
                
                # Clear batch
                batch_examples.clear()
                batch_indices.clear()
        
        # Process remaining examples in the last batch
        if batch_examples:
            self._process_batch(
                batch_examples, batch_indices, adapter, all_predictions, 
                all_references, all_latencies, all_errors, all_cached,
                success_count, error_count, _call_generate, _maybe_read_cache, 
                _maybe_write_cache, concurrency, max_retries, request_timeout
            )
        
        total_duration = time.perf_counter() - t0
        latencies = [x for x in all_latencies if x > 0]
        
        # Calculate metrics
        results: Dict[str, Any] = {}
        for m in metrics:
            try:
                results[m.name] = m.compute(all_predictions, all_references)
            except Exception as err:
                results[m.name] = {"error": f"{err}"}

        # Build result payload (similar to original method)
        import datetime as _dt
        
        # Manifest and other metadata (simplified for streaming)
        manifest: Dict[str, Any] = {
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "streaming": True,
            "batch_size": batch_size,
        }
        
        payload: Dict[str, Any] = {
            "task": self.name,
            "dataset": getattr(dataset, "name", dataset.__class__.__name__),
            "size": total_examples,
            "metrics": results,
            "adapter": getattr(adapter, "name", adapter.__class__.__name__),
            "seed": seed,
            "timing": {
                "avg_latency_ms": (sum(latencies) / len(latencies) * 1000.0) if latencies else 0.0,
                "total_seconds": total_duration,
                "throughput_eps": (total_examples / total_duration) if total_duration > 0 else 0.0,
                "request_successes": success_count,
                "request_errors": error_count,
                "error_rate": (error_count / total_examples) if total_examples > 0 else 0.0,
                "cache_hits": cache_stats.hits,
                "cache_misses": cache_stats.misses,
                "cache_hit_rate": cache_stats.hit_rate,
            },
            "error_summary": _summarize_errors(all_errors),
            "manifest": manifest,
        }
        
        if collect_records:
            records: List[Dict[str, Any]] = []
            for i in range(total_examples):
                rec: Dict[str, Any] = {
                    "id": f"example_{i}",  # Simplified ID for streaming
                    "prediction": all_predictions[i] if i < len(all_predictions) else "",
                    "latency_ms": all_latencies[i] * 1000.0 if i < len(all_latencies) else 0.0,
                }
                if i < len(all_errors) and all_errors[i] is not None:
                    rec["error"] = all_errors[i]
                if i < len(all_cached) and all_cached[i]:
                    rec["cached"] = True
                records.append(rec)
            payload["records"] = records
        
        # Close cache
        if cache is not None:
            try:
                cache.close()
            except Exception:
                pass
                
        return payload

    def _process_batch(
        self,
        batch_examples: List[Example],
        batch_indices: List[int],
        adapter: Adapter,
        all_predictions: List[Any],
        all_references: List[Any], 
        all_latencies: List[float],
        all_errors: List[Optional[str]],
        all_cached: List[bool],
        success_count: int,
        error_count: int,
        _call_generate: Callable[[str], str],
        _maybe_read_cache: Callable[[str], Optional[str]],
        _maybe_write_cache: Callable[[str, str], None],
        concurrency: int,
        max_retries: int,
        request_timeout: Optional[float],
    ) -> None:
        """Process a batch of examples efficiently."""
        batch_size = len(batch_examples)
        
        # Extend result lists
        all_predictions.extend([None] * batch_size)
        all_references.extend([None] * batch_size)
        all_latencies.extend([0.0] * batch_size)
        all_errors.extend([None] * batch_size)
        all_cached.extend([False] * batch_size)
        
        base_index = len(all_predictions) - batch_size
        
        if concurrency <= 1:
            # Sequential processing
            for j, ex in enumerate(batch_examples):
                idx = base_index + j
                all_references[idx] = ex.reference
                prompt = self.build_prompt_with_template(ex)
                
                s = time.perf_counter()
                try:
                    cached = _maybe_read_cache(prompt)
                    if cached is not None:
                        raw = cached
                        all_cached[idx] = True
                    else:
                        raw = _call_generate(prompt)
                    e = time.perf_counter()
                    all_latencies[idx] = e - s
                    all_predictions[idx] = self.postprocess(raw)
                except Exception as err:
                    e = time.perf_counter()
                    all_latencies[idx] = e - s
                    error_category = _categorize_error(err)
                    detailed_error = f"[{error_category}] {str(err)}"
                    all_errors[idx] = detailed_error
        else:
            # Concurrent processing for batch
            with ThreadPoolExecutor(max_workers=min(concurrency, batch_size)) as pool:
                futures = []
                for j, ex in enumerate(batch_examples):
                    idx = base_index + j
                    all_references[idx] = ex.reference
                    prompt = self.build_prompt_with_template(ex)
                    
                    def make_job(job_idx: int, pr: str):
                        def _job():
                            s = time.perf_counter()
                            try:
                                cached = _maybe_read_cache(pr)
                                if cached is not None:
                                    raw = cached
                                    cached_flag = True
                                else:
                                    raw = _call_generate(pr)
                                    cached_flag = False
                                e = time.perf_counter()
                                return job_idx, self.postprocess(raw), (e - s), None, cached_flag
                            except Exception as err:
                                e = time.perf_counter()
                                error_category = _categorize_error(err)
                                detailed_error = f"[{error_category}] {str(err)}"
                                return job_idx, "", (e - s), detailed_error, False
                        return _job
                    
                    futures.append(pool.submit(make_job(idx, prompt)))
                
                for fut in as_completed(futures):
                    idx, pred, dur, err, cached_flag = fut.result()
                    all_predictions[idx] = pred
                    all_latencies[idx] = dur
                    all_errors[idx] = err
                    all_cached[idx] = cached_flag
