from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Protocol
from pathlib import Path
import time
import sys
import platform
from importlib.metadata import version as _pkg_version, PackageNotFoundError
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import set_seed, hash_file, retry_call, run_with_timeout


class Adapter(Protocol):
    """Model API adapter."""

    name: str

    def generate(self, prompt: str, **kwargs: Any) -> str:  # sync for simplicity first
        ...


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

    @abstractmethod
    def build_prompt(self, ex: Example) -> str: ...

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

        success_count = 0
        error_count = 0

        def _call_generate(prompt: str) -> str:
            return retry_call(
                lambda: run_with_timeout(lambda: adapter.generate(prompt), request_timeout),
                retries=max_retries,
            )

        t0 = time.perf_counter()
        if max(1, int(concurrency)) <= 1:
            for i, ex in enumerate(examples):
                references[i] = ex.reference
                prompt = self.build_prompt(ex)
                s = time.perf_counter()
                try:
                    raw = _call_generate(prompt)
                    e = time.perf_counter()
                    per_latency[i] = e - s
                    success_count += 1
                    predictions[i] = self.postprocess(raw)
                except Exception as err:  # pragma: no cover - depends on adapter
                    e = time.perf_counter()
                    per_latency[i] = e - s
                    error_count += 1
                    per_error[i] = str(err)
                    predictions[i] = ""
        else:
            with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:  # pragma: no cover
                futures = []
                for i, ex in enumerate(examples):
                    references[i] = ex.reference
                    prompt = self.build_prompt(ex)

                    def make_job(idx: int, pr: str):
                        def _job():
                            s = time.perf_counter()
                            try:
                                raw = _call_generate(pr)
                                e = time.perf_counter()
                                return idx, self.postprocess(raw), (e - s), None
                            except Exception as err:
                                e = time.perf_counter()
                                return idx, "", (e - s), str(err)

                        return _job

                    futures.append(pool.submit(make_job(i, prompt)))
                for fut in as_completed(futures):
                    idx, pred, dur, err = fut.result()
                    predictions[idx] = pred
                    per_latency[idx] = dur
                    if err is None:
                        success_count += 1
                    else:
                        error_count += 1
                        per_error[idx] = err

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
            "adapter": {
                "name": getattr(adapter, "name", adapter.__class__.__name__),
                "class": f"{adapter.__class__.__module__}.{adapter.__class__.__name__}",
            },
            "task": {
                "name": getattr(self, "name", self.__class__.__name__),
                "class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            },
        }

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
            },
            "manifest": manifest,
        }
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
                records.append(rec)
            payload["records"] = records
        return payload
