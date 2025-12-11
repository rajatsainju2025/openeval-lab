"""Pre-built explanation templates for common code patterns.

This module provides template-based explanation generation for
common code patterns like functions, classes, loops, conditionals,
and more. Templates ensure consistent, high-quality explanations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .types import CodeElement, CodeElementType, ExplainLevel


class PatternType(Enum):
    """Types of code patterns that can be explained."""

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    LOOP = "loop"
    CONDITIONAL = "conditional"
    EXCEPTION_HANDLING = "exception_handling"
    COMPREHENSION = "comprehension"
    GENERATOR = "generator"
    DECORATOR = "decorator"
    CONTEXT_MANAGER = "context_manager"
    LAMBDA = "lambda"
    IMPORT = "import"
    ASSIGNMENT = "assignment"
    ASYNC_FUNCTION = "async_function"
    PROPERTY = "property"
    DATACLASS = "dataclass"
    ENUM = "enum"
    PROTOCOL = "protocol"
    TYPE_ALIAS = "type_alias"
    VARIABLE = "variable"


@dataclass
class PatternMatch:
    """Result of pattern matching against code."""

    pattern_type: PatternType
    matched: bool
    confidence: float
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    sub_patterns: List["PatternMatch"] = field(default_factory=list)


@dataclass
class TemplateSection:
    """A section of an explanation template."""

    name: str
    template: str
    required: bool = True
    level: ExplainLevel = ExplainLevel.SUMMARY
    order: int = 0


class ExplanationTemplate(ABC):
    """Base class for explanation templates."""

    pattern_type: PatternType

    @abstractmethod
    def match(self, element: CodeElement) -> PatternMatch:
        """Check if this template matches the code element.

        Args:
            element: Code element to match against.

        Returns:
            PatternMatch with match result and extracted data.
        """
        pass

    @abstractmethod
    def generate(
        self,
        element: CodeElement,
        match: PatternMatch,
        level: ExplainLevel = ExplainLevel.SUMMARY,
    ) -> str:
        """Generate explanation from template.

        Args:
            element: Code element to explain.
            match: Pattern match result.
            level: Explanation detail level.

        Returns:
            Generated explanation string.
        """
        pass


class FunctionTemplate(ExplanationTemplate):
    """Template for explaining functions."""

    pattern_type = PatternType.FUNCTION

    def __init__(self) -> None:
        """Initialize function template."""
        self.sections = {
            ExplainLevel.SUMMARY: [
                TemplateSection(
                    name="purpose",
                    template="The function `{name}` {purpose_description}.",
                    order=1,
                ),
                TemplateSection(
                    name="parameters",
                    template="It takes {param_count} parameter(s): {param_list}.",
                    required=False,
                    order=2,
                ),
                TemplateSection(
                    name="returns",
                    template="It returns {return_description}.",
                    required=False,
                    order=3,
                ),
            ],
            ExplainLevel.DETAILED: [
                TemplateSection(
                    name="purpose",
                    template="## Purpose\n\nThe function `{name}` {purpose_description}.",
                    order=1,
                ),
                TemplateSection(
                    name="parameters",
                    template="## Parameters\n\n{param_details}",
                    order=2,
                ),
                TemplateSection(
                    name="returns",
                    template="## Returns\n\n{return_description}",
                    order=3,
                ),
                TemplateSection(
                    name="implementation",
                    template="## Implementation\n\n{implementation_details}",
                    order=4,
                ),
            ],
            ExplainLevel.EXPERT: [
                TemplateSection(
                    name="purpose",
                    template="## Overview\n\n`{name}`: {purpose_description}",
                    order=1,
                ),
                TemplateSection(
                    name="signature",
                    template="## Signature\n\n```python\n{signature}\n```",
                    order=2,
                ),
                TemplateSection(
                    name="parameters",
                    template="## Parameters\n\n{param_details}",
                    order=3,
                ),
                TemplateSection(
                    name="returns",
                    template="## Return Value\n\n{return_description}",
                    order=4,
                ),
                TemplateSection(
                    name="implementation",
                    template="## Implementation Details\n\n{implementation_details}",
                    order=5,
                ),
                TemplateSection(
                    name="complexity",
                    template="## Complexity Analysis\n\n{complexity_analysis}",
                    order=6,
                ),
                TemplateSection(
                    name="examples",
                    template="## Usage Examples\n\n{examples}",
                    order=7,
                ),
            ],
        }

    def match(self, element: CodeElement) -> PatternMatch:
        """Match function pattern."""
        if element.type != CodeElementType.FUNCTION:
            return PatternMatch(
                pattern_type=self.pattern_type,
                matched=False,
                confidence=0.0,
            )

        # Extract function data
        source = element.source_code
        name = element.name

        # Parse parameters (basic extraction)
        params = self._extract_parameters(source)
        return_type = self._extract_return_type(source)
        docstring = element.docstring or ""

        return PatternMatch(
            pattern_type=self.pattern_type,
            matched=True,
            confidence=1.0,
            extracted_data={
                "name": name,
                "parameters": params,
                "return_type": return_type,
                "docstring": docstring,
                "is_async": source.strip().startswith("async "),
                "has_decorators": "@" in source.split("def")[0] if "def" in source else False,
            },
        )

    def _extract_parameters(self, source: str) -> List[Dict[str, Any]]:
        """Extract parameter information from source."""
        params = []
        try:
            # Simple extraction between ( and )
            start = source.find("(")
            end = source.find(")")
            if start != -1 and end != -1:
                param_str = source[start + 1 : end]
                if param_str.strip():
                    for param in param_str.split(","):
                        param = param.strip()
                        if param and param != "self" and param != "cls":
                            name = param.split(":")[0].split("=")[0].strip()
                            params.append({"name": name, "raw": param})
        except Exception:
            pass
        return params

    def _extract_return_type(self, source: str) -> Optional[str]:
        """Extract return type annotation."""
        try:
            # Look for -> annotation
            if "->" in source:
                start = source.find("->")
                end = source.find(":", start)
                if start != -1 and end != -1:
                    return source[start + 2 : end].strip()
        except Exception:
            pass
        return None

    def generate(
        self,
        element: CodeElement,
        match: PatternMatch,
        level: ExplainLevel = ExplainLevel.SUMMARY,
    ) -> str:
        """Generate function explanation."""
        data = match.extracted_data
        sections = self.sections.get(level, self.sections[ExplainLevel.SUMMARY])

        parts = []
        for section in sorted(sections, key=lambda s: s.order):
            content = self._render_section(section, data, element)
            if content:
                parts.append(content)

        return "\n\n".join(parts)

    def _render_section(
        self, section: TemplateSection, data: Dict[str, Any], element: CodeElement
    ) -> str:
        """Render a template section."""
        try:
            name = data.get("name", element.name)
            params = data.get("parameters", [])

            context = {
                "name": name,
                "purpose_description": self._infer_purpose(name, element.docstring),
                "param_count": len(params),
                "param_list": ", ".join(p["name"] for p in params) if params else "none",
                "param_details": self._format_params_detailed(params),
                "return_description": data.get("return_type") or "the result",
                "implementation_details": self._describe_implementation(element),
                "signature": self._extract_signature(element.source_code),
                "complexity_analysis": "Time: O(n), Space: O(1)",
                "examples": self._generate_examples(name, params),
            }

            return section.template.format(**context)
        except Exception:
            return ""

    def _infer_purpose(self, name: str, docstring: Optional[str]) -> str:
        """Infer function purpose from name and docstring."""
        if docstring:
            # Use first line of docstring
            first_line = docstring.strip().split("\n")[0]
            return first_line.lower().rstrip(".")

        # Infer from name
        words = []
        current = ""
        for char in name:
            if char == "_":
                if current:
                    words.append(current)
                current = ""
            elif char.isupper():
                if current:
                    words.append(current)
                current = char.lower()
            else:
                current += char
        if current:
            words.append(current)

        if words:
            verb = words[0]
            obj = " ".join(words[1:]) if len(words) > 1 else "data"
            return f"{verb}s {obj}"

        return "performs an operation"

    def _format_params_detailed(self, params: List[Dict[str, Any]]) -> str:
        """Format parameters for detailed view."""
        if not params:
            return "No parameters."

        lines = []
        for param in params:
            lines.append(f"- `{param['name']}`: {param.get('raw', 'value')}")
        return "\n".join(lines)

    def _describe_implementation(self, element: CodeElement) -> str:
        """Describe implementation details."""
        source = element.source_code
        details = []

        if "for " in source:
            details.append("Uses iteration")
        if "while " in source:
            details.append("Uses a while loop")
        if "if " in source:
            details.append("Contains conditional logic")
        if "try:" in source:
            details.append("Includes error handling")
        if "yield" in source:
            details.append("Is a generator function")
        if "await" in source:
            details.append("Uses async/await")

        return ". ".join(details) if details else "Standard implementation."

    def _extract_signature(self, source: str) -> str:
        """Extract function signature."""
        lines = source.strip().split("\n")
        sig_lines = []
        for line in lines:
            sig_lines.append(line)
            if ":" in line and not line.strip().startswith("@"):
                break
        return "\n".join(sig_lines)

    def _generate_examples(self, name: str, params: List[Dict[str, Any]]) -> str:
        """Generate usage examples."""
        param_str = ", ".join(p["name"] for p in params)
        return f"```python\nresult = {name}({param_str})\n```"


class ClassTemplate(ExplanationTemplate):
    """Template for explaining classes."""

    pattern_type = PatternType.CLASS

    def match(self, element: CodeElement) -> PatternMatch:
        """Match class pattern."""
        if element.type != CodeElementType.CLASS:
            return PatternMatch(
                pattern_type=self.pattern_type,
                matched=False,
                confidence=0.0,
            )

        source = element.source_code
        name = element.name

        # Extract class data
        methods = self._extract_methods(source)
        bases = self._extract_bases(source)

        return PatternMatch(
            pattern_type=self.pattern_type,
            matched=True,
            confidence=1.0,
            extracted_data={
                "name": name,
                "methods": methods,
                "bases": bases,
                "docstring": element.docstring or "",
                "is_dataclass": "@dataclass" in source,
                "is_abstract": "ABC" in source or "@abstractmethod" in source,
            },
        )

    def _extract_methods(self, source: str) -> List[str]:
        """Extract method names from source."""
        methods = []
        for line in source.split("\n"):
            line = line.strip()
            if line.startswith("def "):
                name = line[4:].split("(")[0]
                methods.append(name)
        return methods

    def _extract_bases(self, source: str) -> List[str]:
        """Extract base classes."""
        bases = []
        try:
            # Find class declaration
            for line in source.split("\n"):
                if line.strip().startswith("class "):
                    start = line.find("(")
                    end = line.find(")")
                    if start != -1 and end != -1:
                        bases_str = line[start + 1 : end]
                        bases = [b.strip() for b in bases_str.split(",") if b.strip()]
                    break
        except Exception:
            pass
        return bases

    def generate(
        self,
        element: CodeElement,
        match: PatternMatch,
        level: ExplainLevel = ExplainLevel.SUMMARY,
    ) -> str:
        """Generate class explanation."""
        data = match.extracted_data
        name = data.get("name", element.name)
        methods = data.get("methods", [])
        bases = data.get("bases", [])
        docstring = data.get("docstring", "")

        if level == ExplainLevel.SUMMARY:
            base_info = f" inheriting from {', '.join(bases)}" if bases else ""
            method_info = f" with {len(methods)} method(s)" if methods else ""
            return f"The class `{name}`{base_info}{method_info}. {docstring.split('.')[0]}."

        elif level == ExplainLevel.DETAILED:
            parts = [f"## Class: `{name}`"]
            if docstring:
                parts.append(f"\n{docstring}")
            if bases:
                parts.append(f"\n### Inheritance\nExtends: {', '.join(bases)}")
            if methods:
                parts.append("\n### Methods")
                for method in methods:
                    parts.append(f"- `{method}()`")
            return "\n".join(parts)

        else:  # EXPERT
            parts = [f"# Class: `{name}`"]
            if docstring:
                parts.append(f"\n## Description\n{docstring}")
            if bases:
                parts.append(f"\n## Inheritance Hierarchy\n- {' -> '.join(bases + [name])}")
            if methods:
                parts.append("\n## Method Summary")
                public = [m for m in methods if not m.startswith("_")]
                private = [m for m in methods if m.startswith("_") and not m.startswith("__")]
                dunder = [m for m in methods if m.startswith("__")]
                if public:
                    parts.append(f"\n### Public Methods: {', '.join(public)}")
                if private:
                    parts.append(f"\n### Private Methods: {', '.join(private)}")
                if dunder:
                    parts.append(f"\n### Dunder Methods: {', '.join(dunder)}")
            return "\n".join(parts)


class LoopTemplate(ExplanationTemplate):
    """Template for explaining loops."""

    pattern_type = PatternType.LOOP

    def match(self, element: CodeElement) -> PatternMatch:
        """Match loop pattern."""
        source = element.source_code.strip()

        is_for = source.startswith("for ") or "\nfor " in source
        is_while = source.startswith("while ") or "\nwhile " in source

        if not (is_for or is_while):
            return PatternMatch(
                pattern_type=self.pattern_type,
                matched=False,
                confidence=0.0,
            )

        return PatternMatch(
            pattern_type=self.pattern_type,
            matched=True,
            confidence=0.9,
            extracted_data={
                "loop_type": "for" if is_for else "while",
                "has_break": "break" in source,
                "has_continue": "continue" in source,
                "has_else": source.count("else:") > 0,
                "is_nested": source.count("for ") > 1 or source.count("while ") > 1,
            },
        )

    def generate(
        self,
        element: CodeElement,
        match: PatternMatch,
        level: ExplainLevel = ExplainLevel.SUMMARY,
    ) -> str:
        """Generate loop explanation."""
        data = match.extracted_data
        loop_type = data.get("loop_type", "for")

        parts = []
        if loop_type == "for":
            parts.append("This is a `for` loop that iterates over a collection.")
        else:
            parts.append("This is a `while` loop that continues until a condition is met.")

        if data.get("has_break"):
            parts.append("It includes a `break` statement for early exit.")
        if data.get("has_continue"):
            parts.append("It uses `continue` to skip iterations.")
        if data.get("is_nested"):
            parts.append("This is a nested loop structure.")

        return " ".join(parts)


class TemplateRegistry:
    """Registry for explanation templates."""

    def __init__(self) -> None:
        """Initialize template registry."""
        self._templates: Dict[PatternType, ExplanationTemplate] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default templates."""
        self.register(FunctionTemplate())
        self.register(ClassTemplate())
        self.register(LoopTemplate())

    def register(self, template: ExplanationTemplate) -> None:
        """Register a template.

        Args:
            template: Template to register.
        """
        self._templates[template.pattern_type] = template

    def get(self, pattern_type: PatternType) -> Optional[ExplanationTemplate]:
        """Get a template by pattern type.

        Args:
            pattern_type: Pattern type to get template for.

        Returns:
            Template or None if not found.
        """
        return self._templates.get(pattern_type)

    def match_all(self, element: CodeElement) -> List[PatternMatch]:
        """Match all templates against an element.

        Args:
            element: Code element to match.

        Returns:
            List of pattern matches.
        """
        matches = []
        for template in self._templates.values():
            match = template.match(element)
            if match.matched:
                matches.append(match)
        return sorted(matches, key=lambda m: m.confidence, reverse=True)

    def best_match(self, element: CodeElement) -> Optional[PatternMatch]:
        """Find the best matching template.

        Args:
            element: Code element to match.

        Returns:
            Best pattern match or None.
        """
        matches = self.match_all(element)
        return matches[0] if matches else None

    def generate_explanation(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.SUMMARY,
        pattern_type: Optional[PatternType] = None,
    ) -> Optional[str]:
        """Generate explanation using templates.

        Args:
            element: Code element to explain.
            level: Explanation detail level.
            pattern_type: Specific pattern type to use.

        Returns:
            Generated explanation or None.
        """
        if pattern_type:
            template = self.get(pattern_type)
            if template:
                match = template.match(element)
                if match.matched:
                    return template.generate(element, match, level)
            return None

        match = self.best_match(element)
        if match:
            template = self.get(match.pattern_type)
            if template:
                return template.generate(element, match, level)
        return None


# Global template registry
_global_template_registry: Optional[TemplateRegistry] = None


def get_template_registry() -> TemplateRegistry:
    """Get the global template registry.

    Returns:
        Global TemplateRegistry instance.
    """
    global _global_template_registry
    if _global_template_registry is None:
        _global_template_registry = TemplateRegistry()
    return _global_template_registry


def generate_from_template(
    element: CodeElement,
    level: ExplainLevel = ExplainLevel.SUMMARY,
) -> Optional[str]:
    """Generate explanation from templates.

    Args:
        element: Code element to explain.
        level: Explanation detail level.

    Returns:
        Generated explanation or None.
    """
    registry = get_template_registry()
    return registry.generate_explanation(element, level)
