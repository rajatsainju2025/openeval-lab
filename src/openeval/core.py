from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Protocol


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

    def evaluate(self, adapter: Adapter, dataset: Dataset, metrics: List[Metric]) -> Dict[str, Any]:
        predictions: List[Any] = []
        references: List[Any] = []
        for ex in iter(dataset):
            prompt = self.build_prompt(ex)
            raw = adapter.generate(prompt)
            pred = self.postprocess(raw)
            predictions.append(pred)
            references.append(ex.reference)
        results: Dict[str, Any] = {}
        for m in metrics:
            results[m.name] = m.compute(predictions, references)
        return {
            "task": self.name,
            "dataset": getattr(dataset, "name", dataset.__class__.__name__),
            "size": len(predictions),
            "metrics": results,
        }
