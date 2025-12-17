"""Code summarizer for generating concise code summaries.

This module provides functionality to generate various levels of code
summaries, from single-line descriptions to detailed overviews.

Example:
    >>> from openeval.explainers import CodeSummarizer, summarize_code
    >>> code = '''
    ... def fibonacci(n):
    ...     if n <= 1:
    ...         return n
    ...     return fibonacci(n-1) + fibonacci(n-2)
    ... '''
    >>> summary = summarize_code(code)
    >>> print(summary.one_liner)
"""

from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SummaryLevel(Enum):
    """Levels of summary detail."""

    ONE_LINER = "one_liner"
    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


class ElementType(Enum):
    """Types of code elements."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    DECORATOR = "decorator"


@dataclass
class CodeElement:
    """A summarizable code element."""

    element_type: ElementType
    name: str
    code: str
    line_start: int
    line_end: int
    docstring: str | None = None
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ElementSummary:
    """Summary of a single code element."""

    element: CodeElement
    one_liner: str
    brief: str
    detailed: str
    key_points: list[str]
    complexity_indicator: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeSummary:
    """Complete code summary."""

    code: str
    one_liner: str
    brief: str
    standard: str
    detailed: str
    comprehensive: str
    elements: list[ElementSummary]
    statistics: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

    def get_level(self, level: SummaryLevel) -> str:
        """Get summary at specified level."""
        level_map = {
            SummaryLevel.ONE_LINER: self.one_liner,
            SummaryLevel.BRIEF: self.brief,
            SummaryLevel.STANDARD: self.standard,
            SummaryLevel.DETAILED: self.detailed,
            SummaryLevel.COMPREHENSIVE: self.comprehensive,
        }
        return level_map.get(level, self.standard)


@dataclass
class SummaryOptions:
    """Options for code summarization."""

    max_one_liner_length: int = 100
    max_brief_length: int = 300
    include_statistics: bool = True
    include_complexity: bool = True
    language: str = "python"
    style: str = "technical"  # technical, simple, educational


class SummaryStrategy(ABC):
    """Abstract base class for summarization strategies."""

    @abstractmethod
    def summarize_element(self, element: CodeElement, options: SummaryOptions) -> ElementSummary:
        """Summarize a code element."""
        pass

    @abstractmethod
    def summarize_code(
        self, code: str, elements: list[CodeElement], options: SummaryOptions
    ) -> CodeSummary:
        """Summarize complete code."""
        pass


class RuleBasedSummaryStrategy(SummaryStrategy):
    """Rule-based summarization strategy."""

    def __init__(self) -> None:
        """Initialize the strategy."""
        self._patterns: dict[ElementType, list[tuple[str, str]]] = {
            ElementType.FUNCTION: [
                (r"def\s+get_\w+", "Retrieves"),
                (r"def\s+set_\w+", "Sets"),
                (r"def\s+is_\w+", "Checks if"),
                (r"def\s+has_\w+", "Checks whether"),
                (r"def\s+create_\w+", "Creates"),
                (r"def\s+delete_\w+", "Deletes"),
                (r"def\s+update_\w+", "Updates"),
                (r"def\s+validate_\w+", "Validates"),
                (r"def\s+parse_\w+", "Parses"),
                (r"def\s+format_\w+", "Formats"),
                (r"def\s+convert_\w+", "Converts"),
                (r"def\s+calculate_\w+", "Calculates"),
                (r"def\s+process_\w+", "Processes"),
                (r"def\s+load_\w+", "Loads"),
                (r"def\s+save_\w+", "Saves"),
            ],
            ElementType.CLASS: [
                (r"class\s+\w*Manager", "Manages"),
                (r"class\s+\w*Handler", "Handles"),
                (r"class\s+\w*Factory", "Creates instances of"),
                (r"class\s+\w*Builder", "Builds"),
                (r"class\s+\w*Validator", "Validates"),
                (r"class\s+\w*Parser", "Parses"),
                (r"class\s+\w*Formatter", "Formats"),
                (r"class\s+\w*Adapter", "Adapts"),
                (r"class\s+\w*Strategy", "Implements strategy for"),
            ],
        }

    def summarize_element(self, element: CodeElement, options: SummaryOptions) -> ElementSummary:
        """Summarize a code element."""
        one_liner = self._generate_one_liner(element, options)
        brief = self._generate_brief(element, options)
        detailed = self._generate_detailed(element, options)
        key_points = self._extract_key_points(element)
        complexity = self._assess_complexity(element)

        return ElementSummary(
            element=element,
            one_liner=one_liner,
            brief=brief,
            detailed=detailed,
            key_points=key_points,
            complexity_indicator=complexity,
        )

    def summarize_code(
        self, code: str, elements: list[CodeElement], options: SummaryOptions
    ) -> CodeSummary:
        """Summarize complete code."""
        element_summaries = [self.summarize_element(elem, options) for elem in elements]

        # Generate summaries at different levels
        one_liner = self._generate_code_one_liner(elements, options)
        brief = self._generate_code_brief(element_summaries, options)
        standard = self._generate_code_standard(element_summaries, options)
        detailed = self._generate_code_detailed(element_summaries, options)
        comprehensive = self._generate_code_comprehensive(code, element_summaries, options)

        statistics = self._calculate_statistics(code, elements)

        return CodeSummary(
            code=code,
            one_liner=one_liner,
            brief=brief,
            standard=standard,
            detailed=detailed,
            comprehensive=comprehensive,
            elements=element_summaries,
            statistics=statistics,
        )

    def _generate_one_liner(self, element: CodeElement, options: SummaryOptions) -> str:
        """Generate one-line summary."""
        # Try docstring first
        if element.docstring:
            first_line = element.docstring.split("\n")[0].strip()
            if len(first_line) <= options.max_one_liner_length:
                return first_line

        # Use pattern matching
        for pattern, prefix in self._patterns.get(element.element_type, []):
            if re.search(pattern, element.code):
                return f"{prefix} {self._humanize_name(element.name)}"

        # Default based on element type
        if element.element_type == ElementType.FUNCTION:
            return f"Function that handles {self._humanize_name(element.name)}"
        elif element.element_type == ElementType.CLASS:
            return f"Class representing {self._humanize_name(element.name)}"
        elif element.element_type == ElementType.METHOD:
            return f"Method for {self._humanize_name(element.name)}"

        return f"{element.element_type.value.title()}: {element.name}"

    def _generate_brief(self, element: CodeElement, options: SummaryOptions) -> str:
        """Generate brief summary."""
        parts = [self._generate_one_liner(element, options)]

        # Add parameter info for functions
        if element.element_type in [ElementType.FUNCTION, ElementType.METHOD]:
            params = self._extract_parameters(element.code)
            if params:
                parts.append(f"Takes {len(params)} parameter(s): {', '.join(params[:3])}")
                if len(params) > 3:
                    parts[-1] += f" and {len(params) - 3} more"

        # Add return info
        if "->" in element.code:
            return_match = re.search(r"->\s*(\w+(?:\[.*?\])?)", element.code)
            if return_match:
                parts.append(f"Returns: {return_match.group(1)}")

        return ". ".join(parts)

    def _generate_detailed(self, element: CodeElement, options: SummaryOptions) -> str:
        """Generate detailed summary."""
        sections = []

        # Purpose
        sections.append(f"**Purpose**: {self._generate_one_liner(element, options)}")

        # Parameters
        if element.element_type in [ElementType.FUNCTION, ElementType.METHOD]:
            params = self._extract_parameters(element.code)
            if params:
                sections.append(f"**Parameters**: {', '.join(params)}")

        # Return type
        if "->" in element.code:
            return_match = re.search(r"->\s*(\w+(?:\[.*?\])?)", element.code)
            if return_match:
                sections.append(f"**Returns**: {return_match.group(1)}")

        # Docstring
        if element.docstring:
            sections.append(f"**Documentation**: {element.docstring[:200]}...")

        # Complexity
        complexity = self._assess_complexity(element)
        sections.append(f"**Complexity**: {complexity}")

        return "\n".join(sections)

    def _extract_key_points(self, element: CodeElement) -> list[str]:
        """Extract key points from element."""
        points = []

        # Check for async
        if "async def" in element.code:
            points.append("Asynchronous operation")

        # Check for decorators
        decorators = re.findall(r"@(\w+)", element.code)
        if decorators:
            points.append(f"Uses decorators: {', '.join(decorators[:3])}")

        # Check for exception handling
        if "try:" in element.code and "except" in element.code:
            points.append("Includes error handling")

        # Check for type hints
        if ": " in element.code and "->" in element.code:
            points.append("Fully type-annotated")

        # Check for yield
        if "yield" in element.code:
            points.append("Generator function")

        return points

    def _assess_complexity(self, element: CodeElement) -> str:
        """Assess element complexity."""
        lines = element.code.count("\n") + 1

        # Count complexity indicators
        complexity_score = 0
        complexity_score += element.code.count("if ") * 1
        complexity_score += element.code.count("for ") * 2
        complexity_score += element.code.count("while ") * 2
        complexity_score += element.code.count("try:") * 1
        complexity_score += element.code.count("except") * 1
        complexity_score += element.code.count("lambda") * 1

        if complexity_score <= 2 and lines <= 10:
            return "Simple"
        elif complexity_score <= 5 and lines <= 30:
            return "Moderate"
        elif complexity_score <= 10 and lines <= 50:
            return "Complex"
        else:
            return "Very Complex"

    def _humanize_name(self, name: str) -> str:
        """Convert name to human-readable format."""
        # Handle snake_case
        name = name.replace("_", " ")
        # Handle camelCase
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
        return name.lower()

    def _extract_parameters(self, code: str) -> list[str]:
        """Extract parameter names from function definition."""
        match = re.search(r"def\s+\w+\s*\(([^)]*)\)", code)
        if not match:
            return []

        params_str = match.group(1)
        params = []
        for param in params_str.split(","):
            param = param.strip()
            if param and param != "self" and param != "cls":
                # Extract just the name
                param_name = param.split(":")[0].split("=")[0].strip()
                if param_name:
                    params.append(param_name)

        return params

    def _generate_code_one_liner(self, elements: list[CodeElement], options: SummaryOptions) -> str:
        """Generate one-liner for complete code."""
        classes = [e for e in elements if e.element_type == ElementType.CLASS]
        functions = [e for e in elements if e.element_type == ElementType.FUNCTION]

        parts = []
        if classes:
            parts.append(f"{len(classes)} class(es)")
        if functions:
            parts.append(f"{len(functions)} function(s)")

        if not parts:
            return "Code module"

        return f"Module containing {' and '.join(parts)}"

    def _generate_code_brief(self, summaries: list[ElementSummary], options: SummaryOptions) -> str:
        """Generate brief summary for code."""
        if not summaries:
            return "Empty or unanalyzable code."

        lines = [self._generate_code_one_liner([s.element for s in summaries], options)]

        # Add top elements
        for summary in summaries[:3]:
            lines.append(f"- {summary.one_liner}")

        if len(summaries) > 3:
            lines.append(f"- ... and {len(summaries) - 3} more elements")

        return "\n".join(lines)

    def _generate_code_standard(
        self, summaries: list[ElementSummary], options: SummaryOptions
    ) -> str:
        """Generate standard summary for code."""
        sections = []

        # Group by type
        by_type: dict[ElementType, list[ElementSummary]] = {}
        for summary in summaries:
            etype = summary.element.element_type
            if etype not in by_type:
                by_type[etype] = []
            by_type[etype].append(summary)

        # Classes
        if ElementType.CLASS in by_type:
            sections.append("## Classes")
            for summary in by_type[ElementType.CLASS]:
                sections.append(f"- **{summary.element.name}**: {summary.one_liner}")

        # Functions
        if ElementType.FUNCTION in by_type:
            sections.append("## Functions")
            for summary in by_type[ElementType.FUNCTION]:
                sections.append(f"- **{summary.element.name}**: {summary.one_liner}")

        return "\n".join(sections)

    def _generate_code_detailed(
        self, summaries: list[ElementSummary], options: SummaryOptions
    ) -> str:
        """Generate detailed summary for code."""
        sections = []

        for summary in summaries:
            sections.append(f"### {summary.element.name}")
            sections.append(summary.detailed)
            if summary.key_points:
                sections.append("**Key Points**:")
                for point in summary.key_points:
                    sections.append(f"- {point}")
            sections.append("")

        return "\n".join(sections)

    def _generate_code_comprehensive(
        self, code: str, summaries: list[ElementSummary], options: SummaryOptions
    ) -> str:
        """Generate comprehensive summary."""
        elements = [s.element for s in summaries]
        stats = self._calculate_statistics(code, elements)

        sections = [
            "# Code Summary",
            "",
            "## Overview",
            self._generate_code_brief(summaries, options),
            "",
            "## Statistics",
            f"- Total lines: {stats['total_lines']}",
            f"- Code lines: {stats['code_lines']}",
            f"- Classes: {stats['class_count']}",
            f"- Functions: {stats['function_count']}",
            "",
            "## Detailed Analysis",
            self._generate_code_detailed(summaries, options),
        ]

        return "\n".join(sections)

    def _calculate_statistics(self, code: str, elements: list[CodeElement]) -> dict[str, Any]:
        """Calculate code statistics."""
        lines = code.split("\n")
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]

        return {
            "total_lines": len(lines),
            "code_lines": len(code_lines),
            "blank_lines": sum(1 for line in lines if not line.strip()),
            "comment_lines": sum(1 for line in lines if line.strip().startswith("#")),
            "class_count": sum(1 for e in elements if e.element_type == ElementType.CLASS),
            "function_count": sum(1 for e in elements if e.element_type == ElementType.FUNCTION),
            "method_count": sum(1 for e in elements if e.element_type == ElementType.METHOD),
        }


class CodeElementExtractor:
    """Extracts code elements from source code."""

    def extract(self, code: str) -> list[CodeElement]:
        """Extract code elements from code.

        Args:
            code: Source code.

        Returns:
            List of code elements.
        """
        elements = []

        try:
            tree = ast.parse(code)
            elements = self._extract_from_ast(tree, code)
        except SyntaxError:
            elements = self._extract_with_regex(code)

        return elements

    def _extract_from_ast(self, tree: ast.Module, code: str) -> list[CodeElement]:
        """Extract elements using AST."""
        elements = []
        lines = code.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                elements.append(
                    CodeElement(
                        element_type=ElementType.CLASS,
                        name=node.name,
                        code=self._get_node_source(node, lines),
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        docstring=ast.get_docstring(node),
                    )
                )
            elif isinstance(node, ast.FunctionDef):
                # Determine if method or function
                is_method = any(
                    isinstance(parent, ast.ClassDef)
                    for parent in ast.walk(tree)
                    if hasattr(parent, "body") and node in getattr(parent, "body", [])
                )
                elements.append(
                    CodeElement(
                        element_type=ElementType.METHOD if is_method else ElementType.FUNCTION,
                        name=node.name,
                        code=self._get_node_source(node, lines),
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        docstring=ast.get_docstring(node),
                    )
                )
            elif isinstance(node, ast.AsyncFunctionDef):
                elements.append(
                    CodeElement(
                        element_type=ElementType.FUNCTION,
                        name=node.name,
                        code=self._get_node_source(node, lines),
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        docstring=ast.get_docstring(node),
                        metadata={"async": True},
                    )
                )

        return elements

    def _get_node_source(self, node: ast.AST, lines: list[str]) -> str:
        """Get source code for an AST node."""
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            start = node.lineno - 1
            end = node.end_lineno if node.end_lineno else start + 1
            return "\n".join(lines[start:end])
        return ""

    def _extract_with_regex(self, code: str) -> list[CodeElement]:
        """Fallback regex extraction."""
        elements = []

        # Extract classes
        for match in re.finditer(
            r"class\s+(\w+)[^:]*:.*?(?=\nclass\s|\ndef\s(?!\s)|\Z)",
            code,
            re.DOTALL,
        ):
            elements.append(
                CodeElement(
                    element_type=ElementType.CLASS,
                    name=match.group(1),
                    code=match.group(0),
                    line_start=code[: match.start()].count("\n") + 1,
                    line_end=code[: match.end()].count("\n") + 1,
                )
            )

        # Extract functions
        for match in re.finditer(
            r"(async\s+)?def\s+(\w+)\s*\([^)]*\)[^:]*:.*?(?=\n(?:async\s+)?def\s|\nclass\s|\Z)",
            code,
            re.DOTALL,
        ):
            elements.append(
                CodeElement(
                    element_type=ElementType.FUNCTION,
                    name=match.group(2),
                    code=match.group(0),
                    line_start=code[: match.start()].count("\n") + 1,
                    line_end=code[: match.end()].count("\n") + 1,
                    metadata={"async": match.group(1) is not None},
                )
            )

        return elements


class CodeSummarizer:
    """Main class for code summarization."""

    def __init__(
        self,
        strategy: SummaryStrategy | None = None,
    ) -> None:
        """Initialize the summarizer.

        Args:
            strategy: Summarization strategy to use.
        """
        self.strategy = strategy or RuleBasedSummaryStrategy()
        self.extractor = CodeElementExtractor()

    def summarize(
        self,
        code: str,
        options: SummaryOptions | None = None,
    ) -> CodeSummary:
        """Generate a summary of the code.

        Args:
            code: Source code to summarize.
            options: Summarization options.

        Returns:
            CodeSummary with summaries at all levels.
        """
        options = options or SummaryOptions()
        elements = self.extractor.extract(code)
        return self.strategy.summarize_code(code, elements, options)

    def summarize_element(
        self,
        code: str,
        element_name: str,
        options: SummaryOptions | None = None,
    ) -> ElementSummary | None:
        """Summarize a specific element.

        Args:
            code: Source code.
            element_name: Name of element to summarize.
            options: Summarization options.

        Returns:
            ElementSummary if found.
        """
        options = options or SummaryOptions()
        elements = self.extractor.extract(code)

        for element in elements:
            if element.name == element_name:
                return self.strategy.summarize_element(element, options)

        return None

    def get_one_liner(self, code: str) -> str:
        """Get a one-line summary.

        Args:
            code: Source code.

        Returns:
            One-line summary.
        """
        summary = self.summarize(code)
        return summary.one_liner


# Global instance
_code_summarizer: CodeSummarizer | None = None


def get_code_summarizer() -> CodeSummarizer:
    """Get the global code summarizer.

    Returns:
        The global CodeSummarizer instance.
    """
    global _code_summarizer
    if _code_summarizer is None:
        _code_summarizer = CodeSummarizer()
    return _code_summarizer


def reset_code_summarizer() -> None:
    """Reset the global code summarizer."""
    global _code_summarizer
    _code_summarizer = None


def create_code_summarizer(
    strategy: SummaryStrategy | None = None,
) -> CodeSummarizer:
    """Create a new code summarizer.

    Args:
        strategy: Summarization strategy.

    Returns:
        New CodeSummarizer instance.
    """
    return CodeSummarizer(strategy=strategy)


def summarize_code(
    code: str,
    level: SummaryLevel = SummaryLevel.STANDARD,
    **kwargs: Any,
) -> str:
    """Summarize code at the specified level.

    Args:
        code: Source code.
        level: Summary detail level.
        **kwargs: Additional options.

    Returns:
        Summary string.
    """
    options = SummaryOptions(**kwargs) if kwargs else None
    summary = get_code_summarizer().summarize(code, options)
    return summary.get_level(level)


def get_code_one_liner(code: str) -> str:
    """Get a one-line summary of code.

    Args:
        code: Source code.

    Returns:
        One-line summary.
    """
    return get_code_summarizer().get_one_liner(code)


def create_summary_options(
    max_one_liner_length: int = 100,
    include_statistics: bool = True,
    **kwargs: Any,
) -> SummaryOptions:
    """Create summary options.

    Args:
        max_one_liner_length: Maximum one-liner length.
        include_statistics: Whether to include statistics.
        **kwargs: Additional options.

    Returns:
        SummaryOptions instance.
    """
    return SummaryOptions(
        max_one_liner_length=max_one_liner_length,
        include_statistics=include_statistics,
        **kwargs,
    )
