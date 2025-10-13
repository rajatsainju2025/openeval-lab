"""Enhanced configuration system with templates, profiles, and loaders."""

from __future__ import annotations

import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import copy

try:
    import yaml

    HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore
    HAS_YAML = False

try:
    import jinja2

    HAS_JINJA2 = True
except ImportError:
    jinja2 = None  # type: ignore
    HAS_JINJA2 = False


@dataclass
class ConfigTemplate:
    """Configuration template with inheritance and variables."""

    name: str
    description: str = ""
    extends: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def resolve_variables(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve template variables."""
        resolved_config = copy.deepcopy(self.config)
        all_vars = {**self.variables, **variables}

        # Simple variable substitution
        def substitute_vars(obj: Any) -> Any:
            if isinstance(obj, str):
                for key, value in all_vars.items():
                    obj = obj.replace(f"{{{{ {key} }}}}", str(value))
                return obj
            elif isinstance(obj, dict):
                return {k: substitute_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_vars(item) for item in obj]
            return obj

        result = substitute_vars(resolved_config)
        if isinstance(result, dict):
            return result
        return {}


@dataclass
class ConfigProfile:
    """Configuration profile combining multiple templates."""

    name: str
    templates: List[str] = field(default_factory=list)
    overrides: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    environment: str = "development"

    def merge_templates(self, templates: Dict[str, ConfigTemplate]) -> Dict[str, Any]:
        """Merge templates with inheritance."""
        merged_config = {}

        for template_name in self.templates:
            if template_name in templates:
                template = templates[template_name]
                template_config = template.resolve_variables(self.variables)

                # Recursive merge
                merged_config = self._deep_merge(merged_config, template_config)

        # Apply overrides
        merged_config = self._deep_merge(merged_config, self.overrides)

        return merged_config

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = copy.deepcopy(base)

        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)

        return result


class ConfigLoader:
    """Base class for configuration loaders."""

    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from path."""
        raise NotImplementedError

    def save(self, config: Dict[str, Any], path: Union[str, Path]):
        """Save configuration to path."""
        raise NotImplementedError


class JSONConfigLoader(ConfigLoader):
    """JSON configuration loader."""

    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load JSON configuration."""
        with open(path, "r") as f:
            return json.load(f)

    def save(self, config: Dict[str, Any], path: Union[str, Path]):
        """Save JSON configuration."""
        with open(path, "w") as f:
            json.dump(config, f, indent=2)


class YAMLConfigLoader(ConfigLoader):
    """YAML configuration loader."""

    def __init__(self):
        if not HAS_YAML:
            raise ImportError("PyYAML is required for YAML support")

    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML configuration."""
        if yaml is None:
            raise ImportError("PyYAML is required for YAML support")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def save(self, config: Dict[str, Any], path: Union[str, Path]):
        """Save YAML configuration."""
        if yaml is None:
            raise ImportError("PyYAML is required for YAML support")
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)


class TemplateEngine:
    """Template engine for configuration rendering."""

    def __init__(self):
        if HAS_JINJA2 and jinja2 is not None:
            self.env = jinja2.Environment(loader=jinja2.BaseLoader(), autoescape=False)
        else:
            self.env = None

    def render(self, template: str, context: Dict[str, Any]) -> str:
        """Render template with context."""
        if not HAS_JINJA2 or self.env is None:
            # Simple string replacement fallback
            result = template
            for key, value in context.items():
                result = result.replace(f"{{{{ {key} }}}}", str(value))
            return result

        jinja_template = self.env.from_string(template)
        return jinja_template.render(**context)


class EnhancedConfigManager:
    """Enhanced configuration manager with templates and profiles."""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / "config"
        self.templates: Dict[str, ConfigTemplate] = {}
        self.profiles: Dict[str, ConfigProfile] = {}
        self.loaders = {
            ".json": JSONConfigLoader(),
            ".yaml": YAMLConfigLoader() if HAS_YAML else None,
            ".yml": YAMLConfigLoader() if HAS_YAML else None,
        }
        self.template_engine = TemplateEngine()

    def load_template(self, name: str, path: Union[str, Path]):
        """Load template from file."""
        path = Path(path)
        if path.suffix not in self.loaders:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        loader = self.loaders[path.suffix]
        if loader is None:
            raise ImportError(f"Loader for {path.suffix} not available")

        data = loader.load(path)
        template = ConfigTemplate(**data)
        self.templates[name] = template

    def create_profile(self, name: str, templates: List[str], **kwargs):
        """Create configuration profile."""
        profile = ConfigProfile(name=name, templates=templates, **kwargs)
        self.profiles[name] = profile

    def get_config(self, profile_name: str) -> Dict[str, Any]:
        """Get resolved configuration for profile."""
        if profile_name not in self.profiles:
            raise ValueError(f"Profile '{profile_name}' not found")

        profile = self.profiles[profile_name]
        return profile.merge_templates(self.templates)

    def save_profile(self, profile_name: str, path: Union[str, Path]):
        """Save profile configuration to file."""
        config = self.get_config(profile_name)
        path = Path(path)

        if path.suffix not in self.loaders:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        loader = self.loaders[path.suffix]
        if loader is None:
            raise ImportError(f"Loader for {path.suffix} not available")

        loader.save(config, path)


def create_base_template() -> ConfigTemplate:
    """Create base configuration template."""
    return ConfigTemplate(
        name="base",
        description="Base configuration template",
        config={
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "evaluation": {"timeout": 300, "max_retries": 3, "batch_size": 32},
        },
    )


def create_development_profile() -> ConfigProfile:
    """Create development configuration profile."""
    return ConfigProfile(
        name="development",
        templates=["base"],
        overrides={"logging": {"level": "DEBUG"}, "evaluation": {"timeout": 60}},
        environment="development",
    )


def create_production_profile() -> ConfigProfile:
    """Create production configuration profile."""
    return ConfigProfile(
        name="production",
        templates=["base"],
        overrides={
            "logging": {"level": "WARNING"},
            "evaluation": {"timeout": 600, "max_retries": 5, "batch_size": 128},
        },
        environment="production",
    )


__all__ = [
    "ConfigTemplate",
    "ConfigProfile",
    "ConfigLoader",
    "JSONConfigLoader",
    "YAMLConfigLoader",
    "TemplateEngine",
    "EnhancedConfigManager",
    "create_base_template",
    "create_development_profile",
    "create_production_profile",
    "HAS_JINJA2",
]
