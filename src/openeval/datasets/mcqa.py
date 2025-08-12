"""Dataset for multiple choice question answering."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Dict, Any
import json

from ..core import Dataset, Example


@dataclass
class MCQADataset(Dataset):
    """Dataset for multiple choice question answering."""

    path: str
    name: str = "mcqa"

    def __iter__(self) -> Iterator[Example]:
        """Iterate over MCQA examples."""
        p = Path(self.path)

        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_no}: {e}")

                # Extract fields
                example_id = data.get("id", str(line_no))
                question = data.get("question", data.get("input", ""))
                choices = data.get("choices", [])
                answer = data.get("answer", data.get("reference", ""))

                # Create example with proper attributes
                example = MCQAExample(
                    id=str(example_id),
                    input=question,
                    reference=answer,
                    question=question,
                    choices=choices,
                    answer=answer,
                    meta=data,
                )

                yield example


class MCQAExample(Example):
    """Extended example for MCQA with additional fields."""

    def __init__(
        self,
        id: str,
        input: Any,
        reference: Any,
        question: str,
        choices: list,
        answer: str,
        meta: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(id, input, reference, meta)
        self.question = question
        self.choices = choices
        self.answer = answer
