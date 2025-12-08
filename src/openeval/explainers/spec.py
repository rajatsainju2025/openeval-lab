"""Declarative explainer configuration specifications.

Allows defining explainer pipelines in YAML/JSON format.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class ExplainerType(str, Enum):
    """Types of explainers."""

    LLM = "llm"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    RULE_BASED = "rule_based"


class AnalyzerType(str, Enum):
    """Types of code analyzers."""

    AST = "ast"
    SEMANTIC = "semantic"
    COMPLEXITY = "complexity"


class FormatterType(str, Enum):
    """Output format types."""

    TEXT = "text"
    MARKDOWN = "markdown"
    ANSI = "ansi"
    HTML = "html"


class ExplainerConfig(BaseModel):
    """Configuration for a code explainer."""

    name: str = Field(..., description="Name of the explainer")
    type: ExplainerType = Field(..., description="Type of explainer")
    description: Optional[str] = Field(None, description="Description")

    # Explainer-specific config
    model: Optional[str] = Field(None, description="LLM model (for LLM explainers)")
    adapter: Optional[str] = Field(None, description="Adapter to use (e.g., 'openai')")
    cache_enabled: bool = Field(True, description="Enable result caching")
    max_tokens: int = Field(1000, description="Max tokens in explanation")

    # Analysis configuration
    analyzers: List[AnalyzerType] = Field(
        default_factory=lambda: [AnalyzerType.AST, AnalyzerType.SEMANTIC],
        description="Analyzers to use",
    )

    # Explanation settings
    default_level: str = Field("detailed", description="Default explanation level")
    include_code: bool = Field(True, description="Include code in output")
    include_metrics: bool = Field(True, description="Include metrics in output")

    # Output format
    output_format: FormatterType = Field(FormatterType.TEXT, description="Default output format")

    # Additional options
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional options")

    @validator("name")
    def name_not_empty(cls, v):
        """Validate name is not empty."""
        if not v or not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()

    class Config:
        """Pydantic config."""

        use_enum_values = False


class ExplainerPipelineSpec(BaseModel):
    """Specification for an explanation pipeline."""

    name: str = Field(..., description="Pipeline name")
    version: str = Field("1.0", description="Pipeline version")
    description: Optional[str] = Field(None, description="Description")

    # Pipeline components
    explainer: ExplainerConfig = Field(..., description="Main explainer")
    analyzers: List[AnalyzerType] = Field(
        default_factory=lambda: [AnalyzerType.AST],
        description="Code analyzers to run",
    )
    evaluators: List[str] = Field(default_factory=list, description="Evaluators to run")

    # Options
    parallel: bool = Field(False, description="Run in parallel")
    timeout: int = Field(30, description="Timeout in seconds")
    retries: int = Field(3, description="Number of retries on failure")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExplainerPipelineSpec":
        """Create from dictionary."""
        return cls(**data)

    class Config:
        """Pydantic config."""

        use_enum_values = False


class ExplainerRegistry:
    """Registry for predefined explainer configurations."""

    DEFAULT_CONFIGS: Dict[str, ExplainerConfig] = {}

    @classmethod
    def _init_defaults(cls) -> Dict[str, ExplainerConfig]:
        """Initialize default configurations."""
        return {
            "quick": ExplainerConfig(
                name="quick",
                type=ExplainerType.SEMANTIC,
                description="Fast semantic analysis without LLM",
                analyzers=[AnalyzerType.AST],
                include_code=True,
                include_metrics=False,
            ),
            "detailed": ExplainerConfig(
                name="detailed",
                type=ExplainerType.HYBRID,
                description="Detailed analysis with semantic + LLM",
                model="gpt-4",
                adapter="openai",
                analyzers=[
                    AnalyzerType.AST,
                    AnalyzerType.SEMANTIC,
                    AnalyzerType.COMPLEXITY,
                ],
                include_code=True,
                include_metrics=True,
            ),
            "expert": ExplainerConfig(
                name="expert",
                type=ExplainerType.LLM,
                description="Expert-level explanation with full analysis",
                model="gpt-4",
                adapter="openai",
                max_tokens=2000,
                analyzers=[
                    AnalyzerType.AST,
                    AnalyzerType.SEMANTIC,
                    AnalyzerType.COMPLEXITY,
                ],
                include_code=True,
                include_metrics=True,
            ),
        }

    def __init__(self) -> None:
        """Initialize registry."""
        self._configs: Dict[str, ExplainerConfig] = self._init_defaults()

    def register(self, config: ExplainerConfig) -> None:
        """Register an explainer configuration.

        Args:
            config: ExplainerConfig to register.
        """
        self._configs[config.name] = config

    def get(self, name: str) -> Optional[ExplainerConfig]:
        """Get a registered configuration.

        Args:
            name: Configuration name.

        Returns:
            ExplainerConfig or None if not found.
        """
        return self._configs.get(name)

    def list_configs(self) -> List[str]:
        """List all registered configuration names.

        Returns:
            List of configuration names.
        """
        return list(self._configs.keys())

    def remove(self, name: str) -> bool:
        """Remove a configuration.

        Args:
            name: Configuration name.

        Returns:
            True if removed, False if not found.
        """
        if name in self.DEFAULT_CONFIGS:
            return False  # Cannot remove defaults

        if name in self._configs:
            del self._configs[name]
            return True

        return False


class ExplainerTemplates:
    """Pre-defined explanation templates."""

    TEMPLATES = {
        "function": {
            "sections": [
                "purpose",
                "parameters",
                "return_value",
                "algorithm",
                "examples",
            ],
            "focus": "How does this function work?",
        },
        "class": {
            "sections": [
                "purpose",
                "attributes",
                "methods",
                "initialization",
                "usage",
            ],
            "focus": "What does this class represent?",
        },
        "module": {
            "sections": [
                "purpose",
                "main_classes",
                "main_functions",
                "dependencies",
                "usage",
            ],
            "focus": "What is the purpose of this module?",
        },
        "algorithm": {
            "sections": [
                "input",
                "algorithm",
                "output",
                "complexity",
                "proof_correctness",
            ],
            "focus": "How does this algorithm work?",
        },
    }

    @classmethod
    def get_template(cls, element_type: str) -> Optional[Dict[str, Any]]:
        """Get template for element type.

        Args:
            element_type: Type of code element.

        Returns:
            Template dictionary or None.
        """
        return cls.TEMPLATES.get(element_type)

    @classmethod
    def list_templates(cls) -> List[str]:
        """List available templates.

        Returns:
            List of template names.
        """
        return list(cls.TEMPLATES.keys())


# Global registry instance
_explainer_registry = ExplainerRegistry()


def get_explainer_registry() -> ExplainerRegistry:
    """Get the global explainer registry.

    Returns:
        Global ExplainerRegistry instance.
    """
    return _explainer_registry
