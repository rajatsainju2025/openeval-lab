from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Protocol
from pathlib import Path
import time
import sys
import platform
from importlib.metadata import version as _pkg_version, PackageNotFoundError

from .utils import set_seed, hash_file


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
    ) -> Dict[str, Any]:
        set_seed(seed)
        predictions: List[Any] = []
        references: List[Any] = []
        records: List[Dict[str, Any]] = []
        latencies: List[float] = []
        t0 = time.perf_counter()
        for ex in iter(dataset):
            prompt = self.build_prompt(ex)
            s = time.perf_counter()
            raw = adapter.generate(prompt)
            e = time.perf_counter()
            latencies.append(e - s)
            pred = self.postprocess(raw)
            predictions.append(pred)
            references.append(ex.reference)
            if collect_records:
                records.append(
                    {
                        "id": ex.id,
                        "input": ex.input,
                        "reference": ex.reference,
                        "prediction": pred,
                        "latency_ms": (e - s) * 1000.0,
                    }
                )
        total_duration = time.perf_counter() - t0
        results: Dict[str, Any] = {}
        for m in metrics:
            try:
                results[m.name] = m.compute(predictions, references)
            except Exception as err:
                # Record the error string so UIs can show unavailable metrics
                results[m.name] = {"error": f"{err}"}

        # Build manifest for reproducibility
        def _maybe_ver(pkg: str) -> Optional[str]:
            try:
                return _pkg_version(pkg)
            except PackageNotFoundError:
                return None
            except Exception:
                return None

        import datetime as _dt

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
            "size": len(predictions),
            "metrics": results,
            "adapter": getattr(adapter, "name", adapter.__class__.__name__),
            "seed": seed,
            "timing": {
                "avg_latency_ms": (sum(latencies) / len(latencies) * 1000.0) if latencies else 0.0,
                "total_seconds": total_duration,
                "throughput_eps": (
                    (len(predictions) / total_duration) if total_duration > 0 else 0.0
                ),
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
            payload["records"] = records
        return payload
