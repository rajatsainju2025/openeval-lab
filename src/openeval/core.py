from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Protocol
from pathlib import Path
import time

from .utils import set_seed, hash_file


class Adapter(Protocol):
    """Model API adapter."""

    name: str

    def generate(self, prompt: str, **kwargs: Any) -> str:  # sync for simplicity first
        ...


class Metric(Protocol):
    """Evaluation metric interface."""

    name: str

    def compute(self, predictions: Iterable[Any], references: Iterable[Any]) -> Mapping[str, float]:
        ...


@dataclass
class Example:
    id: str
    input: Any
    reference: Any
    meta: Optional[Dict[str, Any]] = None


class Dataset(ABC):
    name: str

    @abstractmethod
    def __iter__(self) -> Iterator[Example]:
        ...

    def __len__(self) -> int:
        return sum(1 for _ in iter(self))


class Task(ABC):
    name: str

    @abstractmethod
    def build_prompt(self, ex: Example) -> str:
        ...

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
            results[m.name] = m.compute(predictions, references)
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
                "throughput_eps": (len(predictions) / total_duration) if total_duration > 0 else 0.0,
            },
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
