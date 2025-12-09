"""Factory for creating explainers with flexible configuration.

Implements factory pattern for clean, configuration-driven explainer creation.
"""

from typing import Any, Dict, Optional

from .base import CodeExplainer
from .cache_manager import CacheManager
from .chain import ChainStrategy, ExplainerChain
from .llm_explainer import LLMCodeExplainer
from .prompt_templates import PromptTemplateManager


class ExplainerFactory:
    """Factory for creating configured CodeExplainer instances.

    Provides clean, configuration-driven approach to explainer creation
    with support for dependency injection and composition.
    """

    def __init__(self) -> None:
        """Initialize the factory."""
        self._explainers: Dict[str, type] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in explainer types."""
        self._explainers["llm"] = LLMCodeExplainer

    def register(self, name: str, explainer_class: type) -> None:
        """Register a custom explainer type.

        Args:
            name: Unique name for the explainer type.
            explainer_class: Class implementing CodeExplainer.

        Raises:
            TypeError: If class doesn't implement CodeExplainer interface.
        """
        if not (isinstance(explainer_class, type) and issubclass(explainer_class, CodeExplainer)):
            raise TypeError(f"{explainer_class} must be a subclass of CodeExplainer")
        self._explainers[name.lower()] = explainer_class

    def create(self, explainer_type: str, **kwargs) -> CodeExplainer:
        """Create an explainer instance.

        Args:
            explainer_type: Type of explainer to create (e.g., 'llm').
            **kwargs: Configuration parameters for the explainer.

        Returns:
            Configured CodeExplainer instance.

        Raises:
            ValueError: If explainer type not found.
            TypeError: If configuration is invalid.
        """
        explainer_type_lower = explainer_type.lower()
        if explainer_type_lower not in self._explainers:
            raise ValueError(
                f"Unknown explainer type: {explainer_type}. "
                f"Available: {list(self._explainers.keys())}"
            )

        explainer_class = self._explainers[explainer_type_lower]
        return explainer_class(**kwargs)

    def create_llm(
        self,
        adapter_name: str = "openai",
        model: str = "gpt-4",
        cache_enabled: bool = True,
        max_tokens: int = 1000,
        cache_manager: Optional[CacheManager] = None,
        template_manager: Optional[PromptTemplateManager] = None,
    ) -> LLMCodeExplainer:
        """Create an LLM explainer with defaults.

        Args:
            adapter_name: LLM adapter to use.
            model: Model name.
            cache_enabled: Enable caching.
            max_tokens: Max tokens in explanation.
            cache_manager: Custom cache manager.
            template_manager: Custom template manager.

        Returns:
            Configured LLMCodeExplainer.
        """
        return LLMCodeExplainer(
            adapter_name=adapter_name,
            model=model,
            cache_enabled=cache_enabled,
            max_tokens=max_tokens,
            cache_manager=cache_manager,
            template_manager=template_manager,
        )

    def create_chain(
        self,
        explainer_configs: list,
        strategy: str = "first_success",
        continue_on_error: bool = True,
    ) -> ExplainerChain:
        """Create a chained explainer.

        Args:
            explainer_configs: List of dicts with 'type' and config kwargs.
            strategy: How to combine results (first_success, aggregate).
            continue_on_error: Continue on error or stop.

        Returns:
            Configured ExplainerChain.

        Example:
            configs = [
                {'type': 'llm', 'model': 'gpt-4'},
                {'type': 'llm', 'model': 'gpt-3.5-turbo'},
            ]
            chain = factory.create_chain(configs)
        """
        # Normalize strategy
        try:
            chain_strategy = ChainStrategy(strategy.lower())
        except ValueError:
            raise ValueError(
                f"Invalid strategy: {strategy}. " f"Valid: {[s.value for s in ChainStrategy]}"
            )

        # Create explainers
        explainers = []
        for config in explainer_configs:
            config_copy = config.copy()
            explainer_type = config_copy.pop("type")
            explainer = self.create(explainer_type, **config_copy)
            explainers.append(explainer)

        # Create chain
        return ExplainerChain(
            explainers=explainers,
            strategy=chain_strategy,
            continue_on_error=continue_on_error,
        )

    def create_from_dict(self, config: Dict[str, Any]) -> CodeExplainer:
        """Create explainer from configuration dict.

        Supports nested chain configuration.

        Args:
            config: Configuration dictionary with 'type' key and other options.

        Returns:
            Configured CodeExplainer.

        Example:
            config = {
                'type': 'chain',
                'explainers': [
                    {'type': 'llm', 'model': 'gpt-4'},
                    {'type': 'llm', 'model': 'gpt-3.5-turbo'},
                ],
                'strategy': 'first_success',
            }
            explainer = factory.create_from_dict(config)
        """
        config_copy = config.copy()
        explainer_type = config_copy.pop("type", None)

        if not explainer_type:
            raise ValueError("Configuration must include 'type' key")

        # Special handling for chains
        if explainer_type.lower() == "chain":
            explainer_configs = config_copy.pop("explainers", [])
            strategy = config_copy.pop("strategy", "first_success")
            continue_on_error = config_copy.pop("continue_on_error", True)

            return self.create_chain(
                explainer_configs,
                strategy=strategy,
                continue_on_error=continue_on_error,
            )

        # Regular explainer creation
        return self.create(explainer_type, **config_copy)

    def list_types(self) -> list:
        """List available explainer types.

        Returns:
            List of available explainer type names.
        """
        return list(self._explainers.keys())

    def get_type_info(self, explainer_type: str) -> Dict[str, Any]:
        """Get information about an explainer type.

        Args:
            explainer_type: Type name.

        Returns:
            Dictionary with type information.

        Raises:
            ValueError: If type not found.
        """
        explainer_type_lower = explainer_type.lower()
        if explainer_type_lower not in self._explainers:
            raise ValueError(f"Unknown explainer type: {explainer_type}")

        explainer_class = self._explainers[explainer_type_lower]
        return {
            "name": explainer_type,
            "class": explainer_class.__name__,
            "module": explainer_class.__module__,
            "doc": explainer_class.__doc__,
        }


# Global factory instance
_global_factory = ExplainerFactory()


def get_explainer_factory() -> ExplainerFactory:
    """Get the global explainer factory instance.

    Returns:
        ExplainerFactory singleton.
    """
    return _global_factory
