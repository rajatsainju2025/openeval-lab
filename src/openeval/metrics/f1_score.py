"""F1 Score metric for evaluation."""

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass
class F1Score:
    """F1 Score metric for precision and recall balanced evaluation."""

    name: str = "f1_score"
    average: str = "macro"  # macro, micro, weighted

    def compute(self, predictions: Iterable[str], references: Iterable[str], **kwargs) -> Mapping[str, float]:
        """Compute F1 score for predictions vs references."""
        preds = list(predictions)
        refs = list(references)

        if len(preds) != len(refs):
            raise ValueError("Predictions and references must have the same length")

        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        count = 0

        for pred, ref in zip(preds, refs):
            pred_tokens = set(str(pred).lower().split())
            ref_tokens = set(str(ref).lower().split())

            precision = 0.0
            recall = 0.0

            if not ref_tokens:
                # If reference is empty, precision/recall are undefined
                if not pred_tokens:
                    f1 = 1.0  # Both empty
                else:
                    f1 = 0.0  # Prediction not empty, reference empty
            else:
                intersection = pred_tokens & ref_tokens
                precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
                recall = len(intersection) / len(ref_tokens)

                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)

            total_f1 += f1
            total_precision += precision
            total_recall += recall
            count += 1

        avg_f1 = total_f1 / count if count > 0 else 0.0
        avg_precision = total_precision / count if count > 0 else 0.0
        avg_recall = total_recall / count if count > 0 else 0.0

        return {
            "f1": avg_f1,
            "precision": avg_precision,
            "recall": avg_recall
        }
