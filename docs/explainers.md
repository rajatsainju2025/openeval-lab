# Code Explainers Documentation

**State-of-the-art Python code explanation system with LLM integration, semantic analysis, and quality evaluation.**

---

## Overview

The OpenEval Code Explainers module provides a modular, extensible system for generating high-quality natural language explanations of Python code. It combines multiple analysis strategies (AST, semantic, complexity) with LLM-powered explanation generation and scientific evaluation of explanation quality.

## Quick Start

### Installation

```bash
pip install openeval-lab
```

### Basic Usage

```python
from openeval.explainers import (
    ExplainLevel,
    PythonASTAnalyzer,
    LLMCodeExplainer,
)

# Read code
with open("mycode.py") as f:
    code = f.read()

# Analyze structure
analyzer = PythonASTAnalyzer()
analysis = analyzer.analyze(code)

# Generate explanations
explainer = LLMCodeExplainer(model="gpt-4")

for element in analysis.elements:
    result = explainer.explain(element, ExplainLevel.DETAILED)
    print(f"{element.name}: {result.explanation}")
```

### CLI Usage

```bash
# Explain a file with detailed analysis
openeval explain-file mycode.py --level detailed --format markdown

# Analyze code structure
openeval analyze mycode.py --detailed

# Evaluate explanation quality
openeval evaluate "This function adds two numbers" --code "def add(a, b): return a + b"
```

---

## Architecture

### Core Components

```
explainers/
├── base.py               # Abstract base classes
├── types.py              # Type definitions
├── ast_analyzer.py       # AST-based analysis
├── semantic_analyzer.py  # Variable/scope tracking
├── complexity_metrics.py # Code metrics
├── formatter.py          # Output formatting
├── llm_explainer.py      # LLM integration
├── evaluation_metrics.py # Quality evaluation
├── spec.py               # Configuration system
└── __init__.py           # Package exports
```

### Analysis Pipeline

```
Python Code
    ↓
AST Analyzer (functions, classes, imports)
    ↓
Semantic Analyzer (variables, scope, data flow)
    ↓
Complexity Metrics (cyclomatic complexity, LOC, etc)
    ↓
LLM Explainer (generate explanation using context)
    ↓
Quality Evaluator (score clarity, completeness, etc)
    ↓
Formatter (text, markdown, HTML, ANSI)
```

---

## Core Interfaces

### CodeAnalyzer

Parses code and extracts structural information.

```python
from openeval.explainers import CodeAnalyzer

class MyAnalyzer(CodeAnalyzer):
    def analyze(self, code: str) -> AnalysisResult:
        """Extract code structure."""
        pass

    def extract_elements(self, code: str) -> List[CodeElement]:
        """Get functions, classes, etc."""
        pass

    def get_dependencies(self, code: str) -> List[str]:
        """Get external dependencies."""
        pass
```

### CodeExplainer

Generates explanations for code elements.

```python
from openeval.explainers import CodeExplainer, ExplainLevel

class MyExplainer(CodeExplainer):
    def explain(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.DETAILED,
        context: Optional[Dict] = None,
    ) -> ExplanationResult:
        """Generate explanation."""
        pass

    def batch_explain(
        self,
        elements: List[CodeElement],
        level: ExplainLevel = ExplainLevel.DETAILED,
    ) -> List[ExplanationResult]:
        """Explain multiple elements."""
        pass
```

### ComplexityAnalyzer

Computes code complexity metrics.

```python
from openeval.explainers import ComplexityAnalyzer

analyzer = ComplexityAnalyzer()
metrics = analyzer.calculate(code)

# Metrics available:
# - cyclomatic_complexity
# - lines_of_code
# - comment_ratio
# - nesting_depth
# - function_count
# - class_count
# - average_function_length
```

### ExplanationEvaluator

Scores explanation quality.

```python
from openeval.explainers import ExplanationEvaluator

evaluator = ExplanationEvaluator()
scores = evaluator.evaluate(explanation, code)

# Scores include:
# - clarity (0-1)
# - completeness (0-1)
# - relevance (0-1)
# - conciseness (0-1)
# - accuracy (0-1)
```

---

## Built-in Implementations

### PythonASTAnalyzer

Extracts Python code structure using Abstract Syntax Trees.

```python
from openeval.explainers.ast_analyzer import PythonASTAnalyzer

analyzer = PythonASTAnalyzer()
analysis = analyzer.analyze(code)

# Available methods:
analysis.elements         # List of CodeElement objects
analysis.dependencies     # External dependencies
analysis.imports          # All imports

# Additional analysis:
analyzer.get_call_graph(code)
analyzer.get_control_flow_complexity(code)
```

### PythonSemanticAnalyzer

Performs semantic analysis including variable tracking and scope analysis.

```python
from openeval.explainers.semantic_analyzer import PythonSemanticAnalyzer

analyzer = PythonSemanticAnalyzer()
analysis = analyzer.analyze(code)

# Get variable analysis
var_analysis = analysis.metadata['variable_analysis']
# - total_variables
# - definitions
# - uses
# - reassignments
# - unused_variables

# Data flow analysis
analyzer.find_data_flow_dependencies(code)
analyzer.get_variable_lifetime(code)
```

### PythonComplexityAnalyzer

Calculates code complexity metrics.

```python
from openeval.explainers.complexity_metrics import PythonComplexityAnalyzer

analyzer = PythonComplexityAnalyzer()
metrics = analyzer.calculate(code)

# Get ratings and indices
complexity_rating = analyzer.rate_complexity(code)  # "Simple" to "Very Complex"
maintainability = analyzer.get_maintainability_index(code)  # 0-100
by_function = analyzer.get_complexity_by_function(code)  # Per-function breakdown
```

### LLMCodeExplainer

Generates explanations using LLMs.

```python
from openeval.explainers.llm_explainer import LLMCodeExplainer

explainer = LLMCodeExplainer(
    adapter_name="openai",
    model="gpt-4",
    cache_enabled=True,
    max_tokens=1000,
)

# Single element
result = explainer.explain(element, ExplainLevel.DETAILED)

# Multiple elements
results = explainer.batch_explain(elements)

# With context
result = explainer.explain_with_context(
    element,
    surrounding_code="...",
    documentation="...",
)

# Cache management
stats = explainer.get_cache_stats()
explainer.reset_cache()
```

### ExplanationQualityEvaluator

Evaluates explanation quality.

```python
from openeval.explainers.evaluation_metrics import ExplanationQualityEvaluator

evaluator = ExplanationQualityEvaluator()

# Individual scores
scores = evaluator.evaluate(explanation, code)
# Returns: clarity, completeness, relevance, conciseness, accuracy

# Overall assessment
overall_score = evaluator.get_overall_score(explanation, code)
rating = evaluator.rate_quality(explanation, code)  # "Poor" to "Excellent"

# Topic coverage
from openeval.explainers.evaluation_metrics import ExplanationCoverageMeasure
coverage = ExplanationCoverageMeasure.get_coverage(explanation)
# Checks: purpose, algorithm, inputs, outputs, complexity, edge_cases
```

### CodeFormatter

Formats explanations for different output media.

```python
from openeval.explainers.formatter import CodeFormatter, OutputFormat

formatter = CodeFormatter()

# Format code block
formatted = formatter.format_code_block(
    code,
    format=OutputFormat.MARKDOWN,
    line_numbers=True,
    max_lines=50,
)

# Format explanation result
formatted = formatter.format_explanation_result(
    result,
    format=OutputFormat.ANSI,
    include_code=True,
)

# With annotations
from openeval.explainers.formatter import LineAnnotation
annotations = [
    LineAnnotation(5, "Important logic here", "highlight"),
    LineAnnotation(12, "Error handling", "info"),
]
formatted = formatter.format_with_annotations(
    code,
    annotations,
    format=OutputFormat.MARKDOWN,
)
```

---

## Configuration System

### Using Predefined Configs

```python
from openeval.explainers.spec import get_explainer_registry

registry = get_explainer_registry()

# Available presets: "quick", "detailed", "expert"
config = registry.get("detailed")

# Register custom
from openeval.explainers.spec import ExplainerConfig, ExplainerType
custom = ExplainerConfig(
    name="my_explainer",
    type=ExplainerType.LLM,
    model="gpt-4",
    cache_enabled=True,
)
registry.register(custom)
```

### Configuration from YAML/JSON

```python
from openeval.explainers.spec import ExplainerPipelineSpec

# Load from dict
config_dict = {
    "name": "myanalyzer",
    "explainer": {
        "name": "llm",
        "type": "llm",
        "model": "gpt-4",
    },
    "analyzers": ["ast", "semantic"],
}
pipeline = ExplainerPipelineSpec.from_dict(config_dict)
```

---

## Explanation Levels

### Summary
Brief 2-3 sentence overview of code functionality.

```python
from openeval.explainers import ExplainLevel
result = explainer.explain(element, ExplainLevel.SUMMARY)
```

### Detailed
Comprehensive explanation including algorithm and key concepts.

### Expert
Advanced explanation with algorithm complexity, edge cases, and best practices.

---

## CLI Commands

### explain-file

```bash
openeval explain-file <file> [options]

Options:
  --level {summary,detailed,expert}  Explanation detail level
  --format {text,markdown,ansi,html} Output format
  --model MODEL                      LLM model to use
  --analyze                          Only analyze, don't explain
  --metrics                          Show complexity metrics
```

### analyze

```bash
openeval analyze <file> [options]

Options:
  --detailed, -d  Show detailed analysis including variable tracking
```

### evaluate

```bash
openeval evaluate <explanation> [options]

Options:
  --code CODE, -c CODE  Code snippet being explained
```

---

## Use Cases

### API Documentation Generation
Generate documentation for code modules automatically.

```python
analyzer = PythonASTAnalyzer()
analysis = analyzer.analyze(module_code)

explainer = LLMCodeExplainer()
for element in analysis.elements:
    explanation = explainer.explain(element, ExplainLevel.DETAILED)
    print(f"## {element.name}\n\n{explanation}")
```

### Code Review Support
Get detailed explanations during code review.

```python
# During PR review, generate explanations for changed functions
evaluator = ExplanationQualityEvaluator()
for element in changed_functions:
    result = explainer.explain(element)
    quality = evaluator.evaluate(result.explanation, element.source_code)
    if quality['clarity'] < 0.7:
        flag_for_improvement(element)
```

### Educational Tool
Explain code to students with multiple levels.

```python
# Summary for quick understanding
summary = explainer.explain(element, ExplainLevel.SUMMARY)

# Detailed for learning
detailed = explainer.explain(element, ExplainLevel.DETAILED)

# Expert for advanced insights
expert = explainer.explain(element, ExplainLevel.EXPERT)
```

### Codebase Onboarding
Help new developers understand codebase.

```python
analyzer = PythonASTAnalyzer()
analysis = analyzer.analyze(codebase)

# Generate explanations for all public APIs
for element in analysis.get_elements_by_type(CodeElementType.FUNCTION):
    if not element.name.startswith('_'):
        explanation = explainer.explain(element)
        # Save to documentation
```

---

## Performance Tips

### Caching
Enable caching to avoid redundant LLM calls:

```python
explainer = LLMCodeExplainer(cache_enabled=True)

# Check cache statistics
stats = explainer.get_cache_stats()
print(f"Cached: {stats['cached_explanations']} explanations")

# Clear cache when needed
explainer.reset_cache()
```

### Batch Processing
Process multiple elements efficiently:

```python
# Batch is more efficient than individual calls
results = explainer.batch_explain(elements)
```

### Parallel Analysis
Run analyzers in parallel for large codebases:

```python
from openeval.explainers.spec import ExplainerPipelineSpec
pipeline = ExplainerPipelineSpec(
    name="parallel",
    parallel=True,  # Enable parallel execution
    timeout=60,
)
```

---

## Extending the System

### Custom Analyzer

```python
from openeval.explainers import CodeAnalyzer, AnalysisResult

class MyLanguageAnalyzer(CodeAnalyzer):
    def analyze(self, code: str) -> AnalysisResult:
        # Implement analysis logic
        pass

    def extract_elements(self, code: str):
        # Implement element extraction
        pass

    def get_dependencies(self, code: str):
        # Implement dependency extraction
        pass

# Register
registry = get_global_registry()
registry.register_analyzer("mylang", MyLanguageAnalyzer)
```

### Custom Explainer

```python
from openeval.explainers import CodeExplainer

class MyExplainer(CodeExplainer):
    def explain(self, element, level, context=None):
        # Implement explanation logic
        pass

    def batch_explain(self, elements, level):
        # Implement batch explanation
        pass

# Register
registry.register_explainer("myexplainer", MyExplainer)
```

---

## Best Practices

1. **Always cache results** when using LLM explainers
2. **Use appropriate detail levels** for your use case
3. **Validate explanation quality** before using in production
4. **Combine analyzers** for comprehensive understanding
5. **Handle errors gracefully** in batch operations
6. **Monitor token usage** for LLM-based explainers

---

## Troubleshooting

### LLM Connection Issues
```python
# Check adapter configuration
try:
    result = explainer.explain(element)
except RuntimeError as e:
    print(f"LLM error: {e}")
    # Fall back to semantic analysis
```

### High Explanation Cost
```python
# Use caching and batch processing
explainer.cache_enabled = True
results = explainer.batch_explain(elements)
```

### Poor Explanation Quality
```python
# Check coverage and quality scores
evaluator = ExplanationQualityEvaluator()
scores = evaluator.evaluate(explanation, code)
print(scores)  # Identify weak areas
```

---

## References

- OpenEval Lab: https://github.com/rajatsainju2025/openeval-lab
- Code Complexity Metrics: https://en.wikipedia.org/wiki/Cyclomatic_complexity
- AST Documentation: https://docs.python.org/3/library/ast.html

---

*Last updated: December 2025*
*OpenEval Lab v0.1.0*
