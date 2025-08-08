from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass
class BERTScore:
    name: str = "bertscore"
    model_type: str = "microsoft/deberta-xlarge-mnli"
    lang: str | None = None
    rescale_with_baseline: bool = True

    def compute(self, predictions: Iterable[str], references: Iterable[str]) -> Mapping[str, float]:
        try:
            from bert_score import score as bertscore  # type: ignore
        except Exception as e:  # pragma: no cover - optional dep
            raise RuntimeError(
                "bert-score not installed. Install with 'pip install -e .[metrics]'"
            ) from e
        cands = [str(p or "") for p in predictions]
        refs = [str(r or "") for r in references]
        P, R, F1 = bertscore(
            cands,
            refs,
            lang=self.lang,
            model_type=self.model_type,
            rescale_with_baseline=self.rescale_with_baseline,
            verbose=False,
        )
        # return average scores
        return {"precision": float(P.mean()), "recall": float(R.mean()), "f1": float(F1.mean())}
