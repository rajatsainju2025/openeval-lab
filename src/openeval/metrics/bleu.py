from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass
class SacreBLEU:
    name: str = "sacrebleu"
    lowercase: bool = False
    tokenize: str | None = None

    def compute(self, predictions: Iterable[str], references: Iterable[str]) -> Mapping[str, float]:
        try:
            import sacrebleu as sb  # type: ignore
        except Exception as e:  # pragma: no cover - optional dep
            raise RuntimeError("sacrebleu not installed. Install with 'pip install -e .[metrics]'") from e
        preds = [str(p or "") for p in predictions]
        refs = [[str(r or "") for r in references]]  # sacrebleu expects list of reference sets
        bleu = sb.corpus_bleu(
            preds,
            refs,
            lowercase=self.lowercase,
            tokenize=self.tokenize or "13a",
        )
        # sacrebleu returns score in [0,100]
        return {"bleu": float(bleu.score) / 100.0}
