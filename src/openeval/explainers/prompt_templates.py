"""Prompt template system for code explanation generation.

Provides pluggable, composable prompt templates for different explanation styles.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional

from .types import CodeElement, ExplainLevel


class PromptStyle(str, Enum):
    """Different prompt styles for code explanation."""

    DIRECT = "direct"  # Straightforward explanation
    SOCRATIC = "socratic"  # Socratic method (questions)
    CHAIN_OF_THOUGHT = "chain_of_thought"  # Step-by-step reasoning
    FEW_SHOT = "few_shot"  # Examples-based
    EXPERT = "expert"  # Technical depth


class PromptTemplate(ABC):
    """Abstract base class for prompt templates."""

    @abstractmethod
    def build(
        self,
        element: CodeElement,
        level: ExplainLevel,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a prompt for the given code element.

        Args:
            element: Code element to explain.
            level: Explanation detail level.
            context: Additional context information.

        Returns:
            Formatted prompt string.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get template name."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get template description."""
        pass


class DirectPromptTemplate(PromptTemplate):
    """Straightforward explanation template."""

    def build(
        self,
        element: CodeElement,
        level: ExplainLevel,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build direct explanation prompt."""
        prompt_parts = []

        # Role setting
        prompt_parts.append("You are an expert Python code explanation assistant.")

        # Task
        level_desc = {
            ExplainLevel.SUMMARY: "concise (2-3 sentences)",
            ExplainLevel.DETAILED: "comprehensive (5-10 sentences)",
            ExplainLevel.EXPERT: "advanced with algorithm analysis",
        }
        prompt_parts.append(
            f"Explain the following {element.type.value} in a {level_desc.get(level, 'detailed')} manner."
        )

        # Code
        prompt_parts.append(f"\nCode to explain:\n```python\n{element.source_code}\n```")

        # Context if available
        if context and context.get("surrounding_code"):
            prompt_parts.append(
                f"\nSurrounding code context:\n```python\n{context['surrounding_code']}\n```"
            )

        if context and context.get("documentation"):
            prompt_parts.append(f"\nRelated documentation:\n{context['documentation']}")

        # Specific guidance for expert level
        if level == ExplainLevel.EXPERT:
            prompt_parts.append(
                "\nInclude: algorithm complexity, edge cases, performance considerations, and best practices."
            )

        prompt_parts.append("\nProvide a clear, educational explanation:")

        return "\n".join(prompt_parts)

    def get_name(self) -> str:
        """Get template name."""
        return "Direct"

    def get_description(self) -> str:
        """Get template description."""
        return "Straightforward, direct explanation style"


class ChainOfThoughtPromptTemplate(PromptTemplate):
    """Chain-of-thought reasoning template."""

    def build(
        self,
        element: CodeElement,
        level: ExplainLevel,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build chain-of-thought prompt."""
        prompt_parts = []

        prompt_parts.append("You are an expert code analyst. Analyze the code step-by-step.")
        prompt_parts.append(
            "\nExplain your reasoning process before providing the final explanation."
        )

        prompt_parts.append(f"\nCode:\n```python\n{element.source_code}\n```")

        prompt_parts.append(
            "\nAnalysis Process:\n"
            "1. First, identify the purpose and primary functionality\n"
            "2. Trace the execution flow and data transformations\n"
            "3. Identify key components and their roles\n"
            "4. Note any important patterns or techniques\n"
            "5. Provide the final explanation"
        )

        if level == ExplainLevel.EXPERT:
            prompt_parts.append("\n6. Analyze complexity and performance implications")

        return "\n".join(prompt_parts)

    def get_name(self) -> str:
        """Get template name."""
        return "Chain of Thought"

    def get_description(self) -> str:
        """Get template description."""
        return "Step-by-step reasoning before explanation"


class SocraticPromptTemplate(PromptTemplate):
    """Socratic method template (educational)."""

    def build(
        self,
        element: CodeElement,
        level: ExplainLevel,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build Socratic method prompt."""
        prompt_parts = []

        prompt_parts.append("You are an educational coding mentor using the Socratic method.")
        prompt_parts.append(
            "Instead of directly explaining, guide the reader to understanding through questions and hints."
        )

        prompt_parts.append(f"\nCode:\n```python\n{element.source_code}\n```")

        prompt_parts.append(
            "\nProvide explanation as a guided discovery:\n"
            "- Start with clarifying questions about what the reader expects\n"
            "- Point out interesting parts worth thinking about\n"
            "- Provide hints rather than direct answers\n"
            "- Guide toward complete understanding step by step"
        )

        return "\n".join(prompt_parts)

    def get_name(self) -> str:
        """Get template name."""
        return "Socratic"

    def get_description(self) -> str:
        """Get template description."""
        return "Educational guided discovery style"


class ExpertPromptTemplate(PromptTemplate):
    """Technical expert analysis template."""

    def build(
        self,
        element: CodeElement,
        level: ExplainLevel,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build expert analysis prompt."""
        prompt_parts = []

        prompt_parts.append(
            "You are a senior software engineer and algorithm expert. Provide technical analysis."
        )

        prompt_parts.append(f"\nCode:\n```python\n{element.source_code}\n```")

        prompt_parts.append(
            "\nProvide expert-level analysis including:\n"
            "- Algorithm: Time and space complexity analysis\n"
            "- Design patterns: Any patterns or principles used\n"
            "- Performance: Potential optimizations and trade-offs\n"
            "- Edge cases: Potential bugs or edge cases\n"
            "- Improvements: How the code could be improved\n"
            "- Best practices: Compliance with coding standards"
        )

        return "\n".join(prompt_parts)

    def get_name(self) -> str:
        """Get template name."""
        return "Expert"

    def get_description(self) -> str:
        """Get template description."""
        return "Advanced technical analysis with complexity and patterns"


class PromptTemplateManager:
    """Manager for prompt templates with registry and selection."""

    def __init__(self) -> None:
        """Initialize template manager."""
        self._templates: Dict[str, PromptTemplate] = {}
        self._default_template = "direct"

        # Register built-in templates
        self.register("direct", DirectPromptTemplate())
        self.register("chain_of_thought", ChainOfThoughtPromptTemplate())
        self.register("socratic", SocraticPromptTemplate())
        self.register("expert", ExpertPromptTemplate())

    def register(self, name: str, template: PromptTemplate) -> None:
        """Register a new template.

        Args:
            name: Unique name for the template.
            template: PromptTemplate instance.
        """
        self._templates[name.lower()] = template

    def get(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name.

        Args:
            name: Template name.

        Returns:
            PromptTemplate or None if not found.
        """
        return self._templates.get(name.lower())

    def set_default(self, name: str) -> None:
        """Set the default template.

        Args:
            name: Template name.

        Raises:
            ValueError: If template not found.
        """
        if name.lower() not in self._templates:
            raise ValueError(f"Template '{name}' not found")
        self._default_template = name.lower()

    def build_prompt(
        self,
        element: CodeElement,
        level: ExplainLevel,
        context: Optional[Dict[str, Any]] = None,
        template_name: Optional[str] = None,
    ) -> str:
        """Build a prompt using specified or default template.

        Args:
            element: Code element to explain.
            level: Explanation level.
            context: Additional context.
            template_name: Specific template to use (uses default if not specified).

        Returns:
            Built prompt string.

        Raises:
            ValueError: If template not found.
        """
        name = template_name or self._default_template
        template = self.get(name)

        if not template:
            raise ValueError(f"Template '{name}' not found")

        return template.build(element, level, context)

    def list_templates(self) -> Dict[str, Dict[str, str]]:
        """List all available templates.

        Returns:
            Dictionary of template names to info (name, description).
        """
        result = {}
        for name, template in self._templates.items():
            result[name] = {
                "name": template.get_name(),
                "description": template.get_description(),
                "is_default": name == self._default_template,
            }
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get template manager statistics.

        Returns:
            Dictionary with statistics.
        """
        return {
            "total_templates": len(self._templates),
            "template_names": list(self._templates.keys()),
            "default_template": self._default_template,
        }
