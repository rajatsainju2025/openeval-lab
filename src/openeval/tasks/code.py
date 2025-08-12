"""Code generation task."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core import Adapter, Example, Task
from ..prompt import PromptTemplate


@dataclass
class CodeGenerationTask(Task):
    """Task for code generation and evaluation."""
    
    name: str = "code_generation"
    
    def __init__(
        self,
        prompt_template: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None
    ):
        """Initialize code generation task."""
        if prompt_template is None:
            # Default template for code generation
            prompt_template = """{{prompt}}"""
        
        super().__init__(prompt_template)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop_sequences = stop_sequences or []
    
    def run(self, adapter: Adapter, ex: Example) -> Dict[str, Any]:
        """Run code generation task."""
        prompt = self.build_prompt_with_template(ex)
        
        # Generate code
        raw_output = adapter.generate(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self.stop_sequences
        )
        
        # Post-process the generated code
        code = self.postprocess(raw_output)
        
        return {
            "prediction": code,
            "raw_output": raw_output,
            "prompt": prompt
        }
    
    def postprocess(self, raw_output: str) -> str:
        """Clean up generated code."""
        code = raw_output.strip()
        
        # Remove common artifacts
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        
        if code.endswith("```"):
            code = code[:-3]
        
        # Remove leading/trailing whitespace
        code = code.strip()
        
        return code


@dataclass
class HumanEvalTask(CodeGenerationTask):
    """Task specifically for HumanEval-style code generation."""
    
    name: str = "humaneval"
    
    def __init__(self, **kwargs):
        """Initialize HumanEval task with appropriate defaults."""
        super().__init__(
            prompt_template="{{prompt}}",
            max_tokens=150,  # HumanEval typically needs shorter completions
            temperature=0.0,  # Deterministic for evaluation
            stop_sequences=["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
            **kwargs
        )
    
    def postprocess(self, raw_output: str) -> str:
        """HumanEval-specific post-processing."""
        code = super().postprocess(raw_output)
        
        # Stop at common function boundaries
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Stop at new function definitions or classes
            stripped = line.strip()
            if (stripped.startswith('def ') or 
                stripped.startswith('class ') or
                stripped.startswith('if __name__')):
                break
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).rstrip()


@dataclass
class CodeCompletionTask(CodeGenerationTask):
    """Task for code completion (filling in missing parts)."""
    
    name: str = "code_completion"
    
    def __init__(self, **kwargs):
        """Initialize code completion task."""
        super().__init__(
            prompt_template="""# Complete the following function:
{{prompt}}

# Your completion:""",
            max_tokens=200,
            temperature=0.1,  # Slightly more diverse for completion
            **kwargs
        )
