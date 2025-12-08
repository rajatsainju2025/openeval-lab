"""Type definitions and data structures for code explainers."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ExplainLevel(str, Enum):
    """Level of detail for code explanations."""

    SUMMARY = "summary"
    DETAILED = "detailed"
    EXPERT = "expert"


class CodeElementType(str, Enum):
    """Types of code elements that can be explained."""

    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    BLOCK = "block"
    EXPRESSION = "expression"
    CONTROL_FLOW = "control_flow"


@dataclass
class CodeElement:
    """Represents a single element of code to be explained."""

    type: CodeElementType
    name: str
    source_code: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Make CodeElement hashable."""
        return hash((self.type, self.name, self.line_start, self.line_end))


@dataclass
class ExplanationResult:
    """Result of explaining a code element."""

    element: CodeElement
    explanation: str
    level: ExplainLevel
    confidence: float  # 0.0 to 1.0
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
    model_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "element": {
                "type": self.element.type.value,
                "name": self.element.name,
                "line_range": f"{self.element.line_start}-{self.element.line_end}",
            },
            "explanation": self.explanation,
            "level": self.level.value,
            "confidence": self.confidence,
            "model": self.model_used,
            "metadata": self.analysis_metadata,
        }


@dataclass
class ComplexityMetrics:
    """Code complexity measurements."""

    cyclomatic_complexity: float
    lines_of_code: int
    comment_ratio: float  # 0.0 to 1.0
    nesting_depth: int
    function_count: int
    class_count: int
    average_function_length: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "lines_of_code": self.lines_of_code,
            "comment_ratio": self.comment_ratio,
            "nesting_depth": self.nesting_depth,
            "function_count": self.function_count,
            "class_count": self.class_count,
            "average_function_length": self.average_function_length,
        }


@dataclass
class AnalysisResult:
    """Result of analyzing code structure."""

    code: str
    elements: List[CodeElement] = field(default_factory=list)
    complexity: Optional[ComplexityMetrics] = None
    dependencies: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_element_by_name(self, name: str) -> Optional[CodeElement]:
        """Get code element by name."""
        for element in self.elements:
            if element.name == name:
                return element
        return None

    def get_elements_by_type(self, elem_type: CodeElementType) -> List[CodeElement]:
        """Get all elements of a specific type."""
        return [e for e in self.elements if e.type == elem_type]
