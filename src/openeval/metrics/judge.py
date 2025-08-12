from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Any, Optional, Tuple, List
from importlib import import_module


@dataclass
class LLMJudge:
    """
    LLM-as-a-judge metric with Balanced Position Calibration.

    Usage in spec (via registry):
      {"name": "llm_judge", "kwargs": {"judge_adapter": "openai-chat", "max_examples": 200}}

    Or via dotted path:
      {"name": "openeval.metrics.judge.LLMJudge", "kwargs": {"judge_adapter": "openeval.adapters.openai.chat_completions.OpenAIChatAdapter", "judge_kwargs": {"model": "gpt-4o-mini"}}}
    """

    name: str = "llm_judge"
    judge_adapter: str = "echo"  # registry key or dotted path
    judge_kwargs: dict[str, Any] = field(default_factory=dict)
    instruction: str = (
        "You are a meticulous evaluator. Given two responses (A and B) to the same prompt,"
        " decide which is better overall in quality, faithfulness, and coherence."
        " Reply with exactly one token: 'A', 'B', or 'Tie'."
    )
    max_examples: Optional[int] = None  # limit judged pairs for speed

    def _load_adapter(self):
        # allow registry short names or dotted path
        from ..registry import lookup

        dotted = lookup("adapter", self.judge_adapter) or self.judge_adapter
        mod_name, cls_name = dotted.replace(":", ".").rsplit(".", 1)
        mod = import_module(mod_name)
        cls = getattr(mod, cls_name)
        return cls(**self.judge_kwargs)

    def _judge_once(self, adapter, prompt: str) -> str:
        try:
            out = adapter.generate(prompt)
        except Exception as e:  # pragma: no cover - network or adapter errors
            return "Tie"
        text = (out or "").strip().upper()
        if "A" in text and "B" not in text:
            return "A"
        if "B" in text and "A" not in text:
            return "B"
        if "TIE" in text:
            return "Tie"
        # fallback: first char heuristic
        if text.startswith("A"):
            return "A"
        if text.startswith("B"):
            return "B"
        return "Tie"

    def _prompts_for(self, prompt: str, pred: str, ref: str) -> Tuple[str, str]:
        common = f"{self.instruction}\n\n" f"Prompt:\n{prompt}\n\n"
        p1 = common + f"Response A:\n{pred}\n\n" + f"Response B:\n{ref}\n\n" + "Answer:"
        p2 = common + f"Response A:\n{ref}\n\n" + f"Response B:\n{pred}\n\n" + "Answer:"
        return p1, p2

    def compute(self, predictions: Iterable[str], references: Iterable[str]) -> Mapping[str, float]:
        # We need the original prompt text to include in judge prompt. We cannot access Task here.
        # As an approximation, we judge prediction vs reference without the original prompt.
        # If reference is empty, we return 0.0.
        preds: List[str] = [str(p or "") for p in predictions]
        refs: List[str] = [str(r or "") for r in references]
        n = len(preds)
        if n == 0:
            return {"win_rate": 0.0, "wins": 0.0, "losses": 0.0, "ties": 0.0}

        adapter = self._load_adapter()
        limit = self.max_examples if self.max_examples is not None else n
        wins = losses = ties = 0
        for i, (p, r) in enumerate(zip(preds, refs)):
            if i >= limit:
                break
            # If no reference, skip
            if not r:
                ties += 1
                continue
            # Build judge prompts without original input; future: allow tasks to pass inputs via records
            base_prompt = ""  # placeholder; could be extended via records in future
            j1, j2 = self._prompts_for(base_prompt, p, r)
            a1 = self._judge_once(adapter, j1)
            a2 = self._judge_once(adapter, j2)
            # Balanced position calibration: aggregate two orders
            if a1 == "A" and a2 == "B":
                # conflicting -> tie
                ties += 1
            elif a1 == "B" and a2 == "A":
                ties += 1
            elif a1 == "A" and a2 == "A":
                wins += 1
            elif a1 == "B" and a2 == "B":
                losses += 1
            else:
                # any Tie in either -> tie
                ties += 1
        total = max(1, wins + losses + ties)
        return {
            "win_rate": wins / total,
            "wins": float(wins),
            "losses": float(losses),
            "ties": float(ties),
        }
