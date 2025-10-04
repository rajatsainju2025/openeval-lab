"""Configuration management system for OpenEval Lab."""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, TypeVar
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod
import copy

try:
    import jinja2

    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False
    jinja2 = None

from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""

    # Execution settings
    max_workers: int = 4
    timeout_seconds: int = 300
    retry_attempts: int = 3

    # Caching settings
    enable_cache: bool = True
    cache_ttl_hours: int = 24

    # Output settings
    output_dir: str = "results"
    save_predictions: bool = True
    save_metrics: bool = True

    # Logging settings
    log_level: str = "INFO"
    log_dir: str = "logs"

    # Statistical settings
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95

    # Bias detection settings
    enable_bias_detection: bool = True
    position_bias_threshold: float = 0.05
    prompt_sensitivity_threshold: float = 0.1


@dataclass
class WebConfig:
    """Configuration for web dashboard."""

    host: str = "localhost"
    port: int = 8000
    debug: bool = False
    cors_origins: Optional[list] = None

    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]


@dataclass
class ConfigTemplate:
    """A configuration template with inheritance support."""

    name: str
    description: str
    extends: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigProfile:
    """A configuration profile combining templates and overrides."""

    name: str
    templates: List[str] = field(default_factory=list)
    overrides: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    environment: str = "development"


class ConfigLoader(ABC):
    """Abstract base class for configuration loaders."""

    @abstractmethod
    def load(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Load configuration from source."""
        pass

    @abstractmethod
    def save(self, config: Dict[str, Any], destination: Union[str, Path]) -> None:
        """Save configuration to destination."""
        pass


class JSONConfigLoader(ConfigLoader):
    """JSON configuration loader."""

    def load(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Load JSON configuration."""
        if isinstance(source, dict):
            return source
        elif isinstance(source, (str, Path)):
            with open(source, "r") as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

    def save(self, config: Dict[str, Any], destination: Union[str, Path]) -> None:
        """Save configuration as JSON."""
        with open(destination, "w") as f:
            json.dump(config, f, indent=2)


class YAMLConfigLoader(ConfigLoader):
    """YAML configuration loader."""

    def load(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Load YAML configuration."""
        if isinstance(source, dict):
            return source
        elif isinstance(source, (str, Path)):
            with open(source, "r") as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

    def save(self, config: Dict[str, Any], destination: Union[str, Path]) -> None:
        """Save configuration as YAML."""
        with open(destination, "w") as f:
            yaml.dump(config, f, default_flow_style=False)


class TemplateEngine:
    """Template rendering engine."""

    def __init__(self):
        if HAS_JINJA2 and jinja2 is not None:
            self.env = jinja2.Environment(loader=jinja2.DictLoader({}), autoescape=False)
        else:
            self.env = None

    def render_template(self, template_str: str, variables: Dict[str, Any]) -> str:
        """Render a template string with variables."""
        if not HAS_JINJA2 or self.env is None:
            # Simple string interpolation fallback
            result = template_str
            for key, value in variables.items():
                result = result.replace(f"{{{{ {key} }}}}", str(value))
            return result

        try:
            template = self.env.from_string(template_str)
            return template.render(**variables)
        except Exception as e:
            logger.warning(f"Template rendering failed: {e}")
            return template_str

    def render_config(self, config: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Render configuration with template variables."""
        rendered = copy.deepcopy(config)

        def render_value(value: Any) -> Any:
            if isinstance(value, str):
                return self.render_template(value, variables)
            elif isinstance(value, dict):
                return {k: render_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [render_value(item) for item in value]
            else:
                return value

        return render_value(rendered)


@dataclass
class OpenEvalConfig:
    """Main configuration for OpenEval Lab."""

    # Sub-configurations
    evaluation: Optional[EvaluationConfig] = None
    web: Optional[WebConfig] = None

    # Global settings
    project_name: str = "openeval-project"
    version: str = "1.0.0"

    # API keys (loaded from environment)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None

    def __post_init__(self):
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.web is None:
            self.web = WebConfig()

        # Load API keys from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")


class ConfigManager:
    """Manages configuration loading, saving, and validation."""

    DEFAULT_CONFIG_PATHS = [
        Path("openeval.yaml"),
        Path("openeval.yml"),
        Path("config/openeval.yaml"),
        Path(".openeval/config.yaml"),
        Path.home() / ".openeval" / "config.yaml",
    ]

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize config manager with optional path."""
        self.config_path = config_path
        self.config: Optional[OpenEvalConfig] = None

    def load_config(self, config_path: Optional[Path] = None) -> OpenEvalConfig:
        """Load configuration from file or defaults."""
        if config_path:
            self.config_path = config_path

        # Try to find config file
        if self.config_path is None:
            self.config_path = self._find_config_file()

        if self.config_path and self.config_path.exists():
            self.config = self._load_from_file(self.config_path)
        else:
            self.config = OpenEvalConfig()

        return self.config

    def save_config(self, config: OpenEvalConfig, path: Optional[Path] = None) -> Path:
        """Save configuration to file."""
        if path is None:
            path = self.config_path or Path("openeval.yaml")

        # Create directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and remove None values
        config_dict = self._clean_dict(asdict(config))

        # Save as YAML
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        return path

    def get_config(self) -> OpenEvalConfig:
        """Get current configuration, loading if necessary."""
        if self.config is None:
            self.load_config()
        assert self.config is not None, "Config should be loaded"
        return self.config

    def update_config(self, updates: Dict[str, Any]) -> OpenEvalConfig:
        """Update configuration with new values."""
        config = self.get_config()

        # Apply updates using dot notation
        for key, value in updates.items():
            self._set_nested_value(config, key, value)

        return config

    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file in default locations."""
        for path in self.DEFAULT_CONFIG_PATHS:
            if path.exists():
                return path
        return None

    def _load_from_file(self, path: Path) -> OpenEvalConfig:
        """Load configuration from YAML or JSON file."""
        with open(path, "r") as f:
            if path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif path.suffix == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")

        return self._dict_to_config(data)

    def _dict_to_config(self, data: Dict[str, Any]) -> OpenEvalConfig:
        """Convert dictionary to OpenEvalConfig object."""

        # Extract sub-configurations
        eval_config = None
        if "evaluation" in data:
            eval_config = EvaluationConfig(**data["evaluation"])

        web_config = None
        if "web" in data:
            web_config = WebConfig(**data["web"])

        # Create main config
        main_data = {k: v for k, v in data.items() if k not in ["evaluation", "web"]}

        config = OpenEvalConfig(**main_data)

        if eval_config:
            config.evaluation = eval_config
        if web_config:
            config.web = web_config

        return config

    def _clean_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None values and empty dicts from dictionary."""
        cleaned = {}
        for k, v in d.items():
            if v is None:
                continue
            elif isinstance(v, dict):
                cleaned_v = self._clean_dict(v)
                if cleaned_v:
                    cleaned[k] = cleaned_v
            else:
                cleaned[k] = v
        return cleaned

    def _set_nested_value(self, obj: Any, key: str, value: Any) -> None:
        """Set nested value using dot notation (e.g., 'evaluation.max_workers')."""
        parts = key.split(".")

        current = obj
        for part in parts[:-1]:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                raise ValueError(f"Invalid config key: {key}")

        final_key = parts[-1]
        if hasattr(current, final_key):
            setattr(current, final_key, value)
        else:
            raise ValueError(f"Invalid config key: {key}")


class EnhancedConfigManager:
    """Enhanced configuration manager with templates and inheritance."""

    def __init__(self):
        self.templates: Dict[str, ConfigTemplate] = {}
        self.profiles: Dict[str, ConfigProfile] = {}
        self.loaders: Dict[str, ConfigLoader] = {
            ".json": JSONConfigLoader(),
            ".yaml": YAMLConfigLoader(),
            ".yml": YAMLConfigLoader(),
        }
        self.template_engine = TemplateEngine()
        self.config_cache: Dict[str, Dict[str, Any]] = {}

    def register_template(self, template: ConfigTemplate):
        """Register a configuration template."""
        self.templates[template.name] = template
        logger.info(f"Registered template: {template.name}")

    def register_profile(self, profile: ConfigProfile):
        """Register a configuration profile."""
        self.profiles[profile.name] = profile
        logger.info(f"Registered profile: {profile.name}")

    def add_loader(self, extension: str, loader: ConfigLoader):
        """Add a custom configuration loader."""
        self.loaders[extension] = loader

    def load_template_from_file(self, file_path: Union[str, Path]) -> ConfigTemplate:
        """Load a template from a file."""
        path = Path(file_path)
        loader = self._get_loader(path.suffix)

        data = loader.load(path)

        template = ConfigTemplate(
            name=data.get("name", path.stem),
            description=data.get("description", ""),
            extends=data.get("extends"),
            config=data.get("config", {}),
            variables=data.get("variables", {}),
            metadata=data.get("metadata", {}),
        )

        self.register_template(template)
        return template

    def load_profile_from_file(self, file_path: Union[str, Path]) -> ConfigProfile:
        """Load a profile from a file."""
        path = Path(file_path)
        loader = self._get_loader(path.suffix)

        data = loader.load(path)

        profile = ConfigProfile(
            name=data.get("name", path.stem),
            templates=data.get("templates", []),
            overrides=data.get("overrides", {}),
            variables=data.get("variables", {}),
            environment=data.get("environment", "development"),
        )

        self.register_profile(profile)
        return profile

    def build_config(
        self, profile_name: str, extra_variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build a complete configuration from a profile."""
        if profile_name not in self.profiles:
            raise ValueError(f"Profile '{profile_name}' not found")

        profile = self.profiles[profile_name]

        # Start with base configuration
        config = {}

        # Apply templates in order
        for template_name in profile.templates:
            if template_name not in self.templates:
                raise ValueError(f"Template '{template_name}' not found")

            template_config = self._resolve_template(template_name)
            self._deep_merge(config, template_config)

        # Apply profile overrides
        self._deep_merge(config, profile.overrides)

        # Collect all variables
        variables = {}
        for template_name in profile.templates:
            template = self.templates[template_name]
            variables.update(template.variables)
        variables.update(profile.variables)
        if extra_variables:
            variables.update(extra_variables)

        # Add environment variables
        variables.update(self._get_environment_variables())

        # Render templates
        config = self.template_engine.render_config(config, variables)

        # Cache the result
        self.config_cache[profile_name] = config

        return config

    def _resolve_template(self, template_name: str) -> Dict[str, Any]:
        """Resolve a template with inheritance."""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")

        template = self.templates[template_name]

        if template.extends:
            # Recursively resolve parent template
            parent_config = self._resolve_template(template.extends)
            config = copy.deepcopy(parent_config)
            self._deep_merge(config, template.config)
        else:
            config = copy.deepcopy(template.config)

        return config

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Deep merge update into base dictionary."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _get_environment_variables(self) -> Dict[str, Any]:
        """Get environment variables with OPENEVAL_ prefix."""
        env_vars = {}
        for key, value in os.environ.items():
            if key.startswith("OPENEVAL_"):
                # Convert OPENEVAL_DATABASE_URL to database_url
                config_key = key[9:].lower()
                env_vars[config_key] = value
        return env_vars

    def _get_loader(self, extension: str) -> ConfigLoader:
        """Get appropriate loader for file extension."""
        if extension not in self.loaders:
            raise ValueError(f"No loader registered for extension: {extension}")
        return self.loaders[extension]

    def save_config(self, config: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        path = Path(file_path)
        loader = self._get_loader(path.suffix)
        loader.save(config, path)

    def validate_config(
        self, config: Dict[str, Any], schema: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Validate configuration against schema."""
        errors = []

        if schema:
            errors.extend(self._validate_against_schema(config, schema))

        # Additional validation rules
        errors.extend(self._validate_config_structure(config))

        return errors

    def _validate_against_schema(self, config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Validate config against JSON schema."""
        # Simplified schema validation
        errors = []

        def validate_section(
            section_config: Dict[str, Any], section_schema: Dict[str, Any], path: str = ""
        ):
            for key, rules in section_schema.items():
                current_path = f"{path}.{key}" if path else key

                if key not in section_config and rules.get("required", False):
                    errors.append(f"Missing required field: {current_path}")
                    continue

                if key in section_config:
                    value = section_config[key]
                    expected_type = rules.get("type")

                    if expected_type == "string" and not isinstance(value, str):
                        errors.append(f"Field {current_path} must be string, got {type(value)}")
                    elif expected_type == "number" and not isinstance(value, (int, float)):
                        errors.append(f"Field {current_path} must be number, got {type(value)}")
                    elif expected_type == "boolean" and not isinstance(value, bool):
                        errors.append(f"Field {current_path} must be boolean, got {type(value)}")
                    elif expected_type == "array" and not isinstance(value, list):
                        errors.append(f"Field {current_path} must be array, got {type(value)}")
                    elif expected_type == "object" and not isinstance(value, dict):
                        errors.append(f"Field {current_path} must be object, got {type(value)}")

        validate_section(config, schema)
        return errors

    def _validate_config_structure(self, config: Dict[str, Any]) -> List[str]:
        """Validate general configuration structure."""
        errors = []

        # Check for required top-level sections
        required_sections = ["logging", "evaluation", "storage"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")

        # Validate logging configuration
        if "logging" in config:
            log_config = config["logging"]
            if "level" in log_config and log_config["level"] not in [
                "DEBUG",
                "INFO",
                "WARNING",
                "ERROR",
            ]:
                errors.append(f"Invalid log level: {log_config['level']}")

        return errors


# Predefined templates and profiles
def create_base_template() -> ConfigTemplate:
    """Create a base configuration template."""
    return ConfigTemplate(
        name="base",
        description="Base configuration template",
        config={
            "logging": {
                "level": "{{ log_level }}",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "{{ log_file }}",
                "max_size": 10485760,  # 10MB
                "backup_count": 5,
            },
            "evaluation": {
                "timeout": 300,
                "max_concurrent": 4,
                "cache_enabled": True,
                "metrics": ["accuracy", "precision", "recall", "f1"],
            },
            "storage": {"type": "local", "path": "{{ data_path }}", "compression": "gzip"},
        },
        variables={"log_level": "INFO", "log_file": "logs/openeval.log", "data_path": "data"},
    )


def create_development_profile() -> ConfigProfile:
    """Create a development configuration profile."""
    return ConfigProfile(
        name="development",
        templates=["base"],
        overrides={"logging": {"level": "DEBUG"}, "evaluation": {"max_concurrent": 2}},
        variables={"log_level": "DEBUG", "data_path": "./dev_data"},
        environment="development",
    )


def create_production_profile() -> ConfigProfile:
    """Create a production configuration profile."""
    return ConfigProfile(
        name="production",
        templates=["base"],
        overrides={
            "logging": {"level": "WARNING"},
            "evaluation": {"timeout": 600, "max_concurrent": 8},
            "storage": {"type": "s3", "bucket": "{{ s3_bucket }}", "region": "{{ aws_region }}"},
        },
        variables={
            "log_level": "WARNING",
            "data_path": "/var/data/openeval",
            "s3_bucket": "openeval-prod-data",
            "aws_region": "us-east-1",
        },
        environment="production",
    )


# Global config manager instance
_global_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get or create global config manager."""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    return _global_config_manager


def get_config() -> OpenEvalConfig:
    """Get current configuration."""
    return get_config_manager().get_config()


def load_config(config_path: Optional[Path] = None) -> OpenEvalConfig:
    """Load configuration from file."""
    return get_config_manager().load_config(config_path)


def save_config(config: OpenEvalConfig, path: Optional[Path] = None) -> Path:
    """Save configuration to file."""
    return get_config_manager().save_config(config, path)


def create_default_config() -> OpenEvalConfig:
    """Create default configuration file."""
    config = OpenEvalConfig()
    config_path = save_config(config)
    print(f"Created default configuration at: {config_path}")
    return config
