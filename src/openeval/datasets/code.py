"""Dataset for code generation evaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, List, Union
import json

from ..core import Dataset, Example


@dataclass
class CodeDataset(Dataset):
    """Dataset for code generation tasks."""
    
    path: str
    name: str = "code"
    
    def __iter__(self) -> Iterator[Example]:
        """Iterate over code examples."""
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
                example_id = data.get("id", data.get("task_id", str(line_no)))
                prompt = data.get("prompt", data.get("input", ""))
                solution = data.get("canonical_solution", data.get("solution", ""))
                test_cases = data.get("test", data.get("tests", []))
                
                # Handle different test case formats
                if isinstance(test_cases, str):
                    test_cases = [test_cases]
                elif not isinstance(test_cases, list):
                    test_cases = []
                
                # Create example
                example = CodeExample(
                    id=str(example_id),
                    input=prompt,
                    reference=test_cases,
                    prompt=prompt,
                    solution=solution,
                    test_cases=test_cases,
                    meta=data
                )
                
                yield example


class CodeExample(Example):
    """Extended example for code generation with additional fields."""
    
    def __init__(
        self,
        id: str,
        input: Any,
        reference: Any,
        prompt: str,
        solution: str,
        test_cases: List[str],
        meta: Optional[Dict[str, Any]] = None
    ):
        super().__init__(id, input, reference, meta)
        self.prompt = prompt
        self.solution = solution
        self.test_cases = test_cases


@dataclass 
class HumanEvalDataset(CodeDataset):
    """Dataset specifically for HumanEval format."""
    
    name: str = "humaneval"
    
    def __iter__(self) -> Iterator[Example]:
        """Iterate over HumanEval examples."""
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
                
                # HumanEval specific fields
                task_id = data.get("task_id", str(line_no))
                prompt = data.get("prompt", "")
                canonical_solution = data.get("canonical_solution", "")
                test = data.get("test", "")
                entry_point = data.get("entry_point", "")
                
                # Create test cases from the test string
                test_cases = [test] if test else []
                
                # Create example
                example = HumanEvalExample(
                    id=task_id,
                    input=prompt,
                    reference=test_cases,
                    prompt=prompt,
                    solution=canonical_solution,
                    test_cases=test_cases,
                    entry_point=entry_point,
                    meta=data
                )
                
                yield example


class HumanEvalExample(CodeExample):
    """HumanEval-specific example with entry point."""
    
    def __init__(
        self,
        id: str,
        input: Any,
        reference: Any,
        prompt: str,
        solution: str,
        test_cases: List[str],
        entry_point: str,
        meta: Optional[Dict[str, Any]] = None
    ):
        super().__init__(id, input, reference, prompt, solution, test_cases, meta)
        self.entry_point = entry_point
