from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Protocol, Union
from pathlib import Path
import time
import sys
import platform
from importlib.metadata import version as _pkg_version, PackageNotFoundError
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import set_seed, hash_file, retry_call, run_with_timeout, hash_prompt
from .cache import PredictionCache, CacheStats
from .prompt import PromptTemplate, build_prompt


def _categorize_error(err: Exception) -> str:
    """Categorize an exception for better error reporting."""
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
    """Summarize errors by category."""
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
    """Model API adapter."""

    name: str

    def generate(self, prompt: str, **kwargs: Any) -> str:  # sync for simplicity first
        ...

    # Optional capability: return text plus token-level logprobs and usage
    def generate_with_logprobs(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - optional
        """
        Optional:
        return {
            "text": str,
            "tokens": [str, ...],
            "logprobs": [float, ...],
            "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
        }
        """
        raise NotImplementedError


class Metric(Protocol):
    """Evaluation metric interface."""

    name: str

    def compute(
        self, predictions: Iterable[Any], references: Iterable[Any]
    ) -> Mapping[str, float]: ...


@dataclass
class Example:
    id: str
    input: Any
    reference: Any
    meta: Optional[Dict[str, Any]] = None


class Dataset(ABC):
    name: str

    @abstractmethod
    def __iter__(self) -> Iterator[Example]: ...

    def __len__(self) -> int:
        return sum(1 for _ in iter(self))


class Task(ABC):
    name: str

    def __init__(self, prompt_template: Optional[Union[str, PromptTemplate]] = None):
        """Initialize task with optional prompt template."""
        self._prompt_template_raw = prompt_template
        if isinstance(prompt_template, str):
            self.prompt_template = PromptTemplate(prompt_template)
        else:
            self.prompt_template = prompt_template

    @abstractmethod
    def build_prompt(self, ex: Example) -> str: ...

    def build_prompt_with_template(self, ex: Example, **extra_vars) -> str:
        """Build prompt using template if available, otherwise fallback to build_prompt."""
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
    ) -> Dict[str, Any]:
        set_seed(seed)
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
            cached = _maybe_read_cache(prompt)
            if cached is not None:
                return cached
            out = retry_call(
                lambda: run_with_timeout(lambda: adapter.generate(prompt), request_timeout),
                retries=max_retries,
            )
            _maybe_write_cache(prompt, out)
            return out

        t0 = time.perf_counter()
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
