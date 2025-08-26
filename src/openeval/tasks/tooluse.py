from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Mapping, Optional, Dict
import time
from pathlib import Path
import sys
import platform
from importlib.metadata import version as _pkg_version, PackageNotFoundError

from ..core import Task, Example, Dataset, Adapter, Metric
from ..agents.base import Agent


@dataclass
class ToolUseExample(Example):
    tools: Optional[List[str]] = None  # allowed tool names for this example


class ToolUseTask(Task):
    name = "tool_use"

    def __init__(self, agent: Agent, prompt_template: Optional[str] = None):
        super().__init__(prompt_template)
        self.agent = agent

    def build_prompt(self, ex: Example) -> str:
        # Not used for agentic tasks; agent consumes raw input
        return str(ex.input)

    def postprocess(self, raw_output: str) -> Any:
        # Not used; agent returns structured result
        return raw_output

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
    ) -> Dict[str, Any]:  # pragma: no cover - exercised via CLI/tests
        # Minimal sequential loop using the agent
        examples: List[Example] = list(iter(dataset))
        predictions: List[Any] = []
        references: List[Any] = []
        per_latency: List[float] = []

        t0 = time.perf_counter()
        for ex in examples:
            s = time.perf_counter()
            res = self.agent.run(ex.input)
            e = time.perf_counter()
            predictions.append(res)
            references.append(ex.reference)
            per_latency.append(e - s)

        total_duration = time.perf_counter() - t0
        latencies = [x for x in per_latency if x > 0]

        results: Dict[str, Any] = {}
        for m in metrics:
            # If metric can accept agent results, it should set attribute
            accepts_agent = getattr(m, "accepts_agent", False)
            preds_for_metric = predictions if accepts_agent else [
                (p.final_answer if hasattr(p, "final_answer") else p) for p in predictions
            ]
            results[m.name] = m.compute(preds_for_metric, references)

        # Build manifest (subset)
        def _maybe_ver(pkg: str) -> Optional[str]:
            try:
                return _pkg_version(pkg)
            except PackageNotFoundError:
                return None
            except Exception:
                return None

        manifest: Dict[str, Any] = {
            "openeval_version": _maybe_ver("openeval-lab"),
            "python": {"version": sys.version.split()[0], "executable": sys.executable},
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "agent": getattr(self.agent, "name", self.agent.__class__.__name__),
        }

        payload: Dict[str, Any] = {
            "task": self.name,
            "dataset": getattr(dataset, "name", dataset.__class__.__name__),
            "size": len([p for p in predictions if p is not None]),
            "metrics": results,
            "adapter": getattr(adapter, "name", adapter.__class__.__name__),
            "timing": {
                "avg_latency_ms": (sum(latencies) / len(latencies) * 1000.0) if latencies else 0.0,
                "total_seconds": total_duration,
            },
            "manifest": manifest,
        }

        if collect_records:
            records: List[Dict[str, Any]] = []
            for ex, pred, lat in zip(examples, predictions, per_latency):
                records.append(
                    {
                        "id": ex.id,
                        "input": ex.input,
                        "reference": ex.reference,
                        "prediction": getattr(pred, "final_answer", str(pred)),
                        "latency_ms": lat * 1000.0,
                    }
                )
            payload["records"] = records

        return payload
