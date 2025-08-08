from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Protocol

from .utils import set_seed


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
        for ex in iter(dataset):
            prompt = self.build_prompt(ex)
            raw = adapter.generate(prompt)
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
                    }
                )
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
        }
        if collect_records:
            payload["records"] = records
        return payload
