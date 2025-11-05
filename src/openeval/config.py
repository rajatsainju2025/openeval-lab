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


"""Advanced unified configuration system with environment-aware configs and validation.

This module provides a comprehensive configuration management system that unifies
CLI flags, YAML configs, environment variables, and secrets into a single coherent
interface with validation, schema enforcement, and environment-specific overrides.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import time
import importlib.util
from .logging import get_logger

HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None

logger = get_logger(__name__)


class ConfigSource(Enum):
    """Configuration source priority order (higher = more priority)."""

    DEFAULT = 1
    FILE = 2
    ENVIRONMENT = 3
    CLI = 4
    OVERRIDE = 5


class Environment(Enum):
    """Environment types for configuration profiles."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ConfigMetadata:
    """Metadata for configuration tracking."""

    source: ConfigSource
    file_path: Optional[str] = None
    env_var: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    validated: bool = False
    hash: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Core evaluation configuration."""

    # Execution settings
    concurrency: int = 4
    timeout: Optional[float] = 300.0
    max_retries: int = 3
    batch_size: int = 10
    streaming: bool = False

    # Caching settings
    cache_enabled: bool = True
    cache_ttl: int = 3600
    cache_dir: str = ".openeval_cache"
    cache_compression: bool = True

    # Resource management
    max_memory_gb: Optional[float] = None
    memory_monitoring: bool = True
    resource_limits: Dict[str, Any] = field(default_factory=dict)

    # Logging and monitoring
    log_level: str = "INFO"
    log_format: str = "structured"
    telemetry_enabled: bool = True
    metrics_endpoint: Optional[str] = None

    # Security settings
    api_key_rotation: bool = False
    secret_encryption: bool = True
    audit_logging: bool = False

    # Statistical settings
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    significance_threshold: float = 0.05
    bias_detection: bool = True

    # Output settings
    output_format: str = "json"
    artifact_compression: bool = True
    detailed_results: bool = False
    export_raw_data: bool = False


@dataclass
class AdapterConfig:
    """Model adapter configuration."""

    # API settings
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    request_timeout: float = 30.0
    rate_limit_rpm: Optional[int] = None

    # Model parameters
    model_name: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Advanced settings
    use_chat_format: bool = False
    system_prompt: Optional[str] = None
    retry_backoff: float = 1.0
    connection_pool_size: int = 10


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    # Data source
    data_path: Optional[str] = None
    data_format: str = "jsonl"
    data_url: Optional[str] = None
    cache_data: bool = True

    # Processing
    shuffle: bool = False
    sample_size: Optional[int] = None
    train_test_split: Optional[float] = None
    preprocessing: List[str] = field(default_factory=list)

    # Validation
    schema_validation: bool = True
    data_quality_checks: bool = True
    missing_data_strategy: str = "skip"


@dataclass
class SecurityConfig:
    """Security configuration settings."""

    enable_encryption: bool = True
    secret_store_type: str = "local"
    audit_log_enabled: bool = True
    audit_log_path: Optional[str] = None
    token_expiry_hours: int = 24
    password_min_length: int = 8
    enable_mfa: bool = False
    rate_limiting_enabled: bool = True
    max_login_attempts: int = 5
    session_timeout_minutes: int = 60
    encryption_key_rotation_days: int = 90
    api_key_encryption: bool = True
    audit_logging: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.password_min_length < 8:
            raise ValueError("Password minimum length must be at least 8 characters")
        if self.token_expiry_hours <= 0:
            raise ValueError("Token expiry hours must be positive")
        if self.session_timeout_minutes <= 0:
            raise ValueError("Session timeout must be positive")


@dataclass
class ObservabilityConfig:
    """Observability and monitoring configuration."""

    # Tracing
    tracing_enabled: bool = False
    tracing_endpoint: Optional[str] = None
    trace_sample_rate: float = 0.1

    # Metrics
    metrics_enabled: bool = True
    metrics_port: int = 8080
    custom_metrics: List[str] = field(default_factory=list)

    # Logging
    structured_logging: bool = True
    log_correlation_id: bool = True
    external_log_endpoint: Optional[str] = None


@dataclass
class UnifiedConfig:
    """Master configuration container."""

    # Environment
    environment: Environment = Environment.DEVELOPMENT
    config_version: str = "1.0"

    # Component configs
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    # Custom extensions
    extensions: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    _metadata: Dict[str, ConfigMetadata] = field(default_factory=dict, init=False)
    _validation_errors: List[str] = field(default_factory=list, init=False)


class ConfigurationError(Exception):
    """Configuration-related error."""

    pass


class ConfigValidator:
    """Advanced configuration validation with schema support."""

    def __init__(self):
        self.validation_rules = {}
        self._load_schemas()

    def _load_schemas(self):
        """Load validation schemas."""
        if HAS_PYDANTIC:
            # Generate Pydantic schemas for validation
            self.validation_rules = {
                "evaluation": self._create_evaluation_schema(),
                "adapter": self._create_adapter_schema(),
                "dataset": self._create_dataset_schema(),
                "security": self._create_security_schema(),
                "observability": self._create_observability_schema(),
            }

    def _create_evaluation_schema(self) -> Optional[Any]:
        """Create Pydantic schema for evaluation config."""
        if not HAS_PYDANTIC:
            return None

        from pydantic import BaseModel, Field, validator

        class EvaluationSchema(BaseModel):
            concurrency: int = Field(ge=1, le=100, description="Number of concurrent workers")
            timeout: Optional[float] = Field(ge=0, description="Timeout in seconds")
            max_retries: int = Field(ge=0, le=10, description="Maximum retry attempts")
            batch_size: int = Field(ge=1, le=1000, description="Batch size for processing")
            cache_ttl: int = Field(ge=60, le=86400, description="Cache TTL in seconds")
            confidence_level: float = Field(
                ge=0.5, le=0.999, description="Statistical confidence level"
            )

            @validator("log_level")
            def validate_log_level(cls, v):
                allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                if v.upper() not in allowed:
                    raise ValueError(f"log_level must be one of {allowed}")
                return v.upper()

            @validator("output_format")
            def validate_output_format(cls, v):
                allowed = ["json", "yaml", "csv", "parquet"]
                if v not in allowed:
                    raise ValueError(f"output_format must be one of {allowed}")
                return v

        return EvaluationSchema

    def _create_adapter_schema(self) -> Optional[Any]:
        """Create Pydantic schema for adapter config."""
        if not HAS_PYDANTIC:
            return None

        from pydantic import BaseModel, Field, validator

        class AdapterSchema(BaseModel):
            temperature: float = Field(ge=0.0, le=2.0, description="Sampling temperature")
            request_timeout: float = Field(ge=1.0, le=300.0, description="Request timeout")
            rate_limit_rpm: Optional[int] = Field(
                ge=1, le=10000, description="Rate limit per minute"
            )

            @validator("api_endpoint")
            def validate_api_endpoint(cls, v):
                if v and not (v.startswith("http://") or v.startswith("https://")):
                    raise ValueError("api_endpoint must be a valid URL")
                return v

        return AdapterSchema

    def _create_dataset_schema(self) -> Optional[Any]:
        """Create Pydantic schema for dataset config."""
        if not HAS_PYDANTIC:
            return None

        from pydantic import BaseModel, Field, validator

        class DatasetSchema(BaseModel):
            data_format: str = Field(description="Data format")
            sample_size: Optional[int] = Field(ge=1, description="Sample size")
            train_test_split: Optional[float] = Field(
                ge=0.1, le=0.9, description="Train/test split ratio"
            )

            @validator("data_format")
            def validate_data_format(cls, v):
                allowed = ["jsonl", "json", "csv", "parquet", "hf", "yaml"]
                if v not in allowed:
                    raise ValueError(f"data_format must be one of {allowed}")
                return v

            @validator("missing_data_strategy")
            def validate_missing_data_strategy(cls, v):
                allowed = ["skip", "fill", "error"]
                if v not in allowed:
                    raise ValueError(f"missing_data_strategy must be one of {allowed}")
                return v

        return DatasetSchema

    def _create_security_schema(self) -> Optional[Any]:
        """Create Pydantic schema for security config."""
        if not HAS_PYDANTIC:
            return None

        from pydantic import BaseModel, Field, validator

        class SecuritySchema(BaseModel):
            secret_store_type: str = Field(description="Secret store type")
            api_key_rotation_days: int = Field(ge=1, le=365, description="API key rotation period")

            @validator("secret_store_type")
            def validate_secret_store_type(cls, v):
                allowed = ["local", "vault", "aws-secrets", "azure-kv", "gcp-secrets"]
                if v not in allowed:
                    raise ValueError(f"secret_store_type must be one of {allowed}")
                return v

        return SecuritySchema

    def _create_observability_schema(self) -> Optional[Any]:
        """Create Pydantic schema for observability config."""
        if not HAS_PYDANTIC:
            return None

        from pydantic import BaseModel, Field

        class ObservabilitySchema(BaseModel):
            trace_sample_rate: float = Field(ge=0.0, le=1.0, description="Trace sampling rate")
            metrics_port: int = Field(ge=1024, le=65535, description="Metrics port")

        return ObservabilitySchema

    def validate_config(self, config_section: str, data: Dict[str, Any]) -> List[str]:
        """Validate a configuration section."""
        errors = []

        if not HAS_PYDANTIC or config_section not in self.validation_rules:
            logger.warning(f"No validation schema available for {config_section}")
            return errors

        schema_class = self.validation_rules[config_section]
        if not schema_class:
            return errors

        try:
            # Validate using Pydantic
            schema_class(**data)
        except Exception as e:
            if HAS_PYDANTIC:
                from pydantic import ValidationError

                if isinstance(e, ValidationError):
                    for error in e.errors():
                        field_path = ".".join(str(loc) for loc in error["loc"])
                        errors.append(f"{config_section}.{field_path}: {error['msg']}")
                else:
                    errors.append(f"{config_section}: Validation error - {str(e)}")
            else:
                errors.append(f"{config_section}: Validation error - {str(e)}")

        return errors


class ConfigManager:
    """Advanced configuration manager with unified config handling."""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path.cwd() / "configs"
        self.validator = ConfigValidator()
        self._config_cache: Dict[str, UnifiedConfig] = {}
        self._file_watchers: Dict[str, float] = {}  # file path -> last modified

        # Environment detection
        self.environment = self._detect_environment()

        logger.info(f"ConfigManager initialized for environment: {self.environment.value}")

    def _detect_environment(self) -> Environment:
        """Detect current environment from various sources."""
        # Check environment variable
        env_name = os.getenv("OPENEVAL_ENV", os.getenv("ENV", "development")).lower()

        # Map common environment names
        env_mapping = {
            "dev": Environment.DEVELOPMENT,
            "development": Environment.DEVELOPMENT,
            "test": Environment.TESTING,
            "testing": Environment.TESTING,
            "stage": Environment.STAGING,
            "staging": Environment.STAGING,
            "prod": Environment.PRODUCTION,
            "production": Environment.PRODUCTION,
        }

        return env_mapping.get(env_name, Environment.DEVELOPMENT)

    def load_config(
        self,
        config_name: Optional[str] = None,
        config_path: Optional[Path] = None,
        cli_overrides: Optional[Dict[str, Any]] = None,
        env_prefix: str = "OPENEVAL",
    ) -> UnifiedConfig:
        """Load configuration from multiple sources with proper precedence."""

        # Determine config file to load
        if config_path:
            config_file = config_path
        elif config_name:
            config_file = self.config_dir / f"{config_name}.yaml"
        else:
            # Use environment-specific default
            config_file = self.config_dir / f"{self.environment.value}.yaml"

        cache_key = str(config_file)

        # Check if config is cached and file hasn't changed
        if cache_key in self._config_cache and self._is_cache_valid(config_file):
            logger.debug(f"Using cached configuration: {config_file}")
            return self._config_cache[cache_key]

        logger.info(f"Loading configuration from: {config_file}")

        # Start with default configuration
        config = UnifiedConfig(environment=self.environment)

        # Load from file if exists
        if config_file.exists():
            config = self._load_from_file(config, config_file)
        else:
            logger.warning(f"Configuration file not found: {config_file}")

        # Apply environment variable overrides
        config = self._apply_env_overrides(config, env_prefix)

        # Apply CLI overrides
        if cli_overrides:
            config = self._apply_cli_overrides(config, cli_overrides)

        # Validate configuration
        self._validate_config(config)

        # Cache the configuration
        self._config_cache[cache_key] = config
        self._file_watchers[str(config_file)] = (
            config_file.stat().st_mtime if config_file.exists() else 0
        )

        return config

    def _load_from_file(self, config: UnifiedConfig, config_file: Path) -> UnifiedConfig:
        """Load configuration from YAML/JSON file."""
        try:
            with open(config_file, "r") as f:
                if config_file.suffix.lower() in [".yaml", ".yml"]:
                    file_data = yaml.safe_load(f)
                elif config_file.suffix.lower() == ".json":
                    file_data = json.load(f)
                else:
                    raise ConfigurationError(
                        f"Unsupported config file format: {config_file.suffix}"
                    )

            if not file_data:
                return config

            # Apply configuration data to config object
            config = self._merge_config_data(config, file_data, ConfigSource.FILE, str(config_file))

            logger.debug(f"Loaded configuration from file: {config_file}")

        except Exception as e:
            raise ConfigurationError(f"Error loading config file {config_file}: {str(e)}")

        return config

    def _apply_env_overrides(self, config: UnifiedConfig, env_prefix: str) -> UnifiedConfig:
        """Apply environment variable overrides."""
        env_vars = {k: v for k, v in os.environ.items() if k.startswith(f"{env_prefix}_")}

        for env_var, value in env_vars.items():
            # Parse environment variable name to config path
            # E.g., OPENEVAL_EVALUATION_CONCURRENCY -> evaluation.concurrency
            config_path = env_var.lower().replace(f"{env_prefix.lower()}_", "").split("_")

            if len(config_path) >= 2:
                section = config_path[0]
                field_name = "_".join(config_path[1:])

                # Convert string value to appropriate type
                typed_value = self._convert_env_value(value)

                # Apply to config
                if hasattr(config, section):
                    section_config = getattr(config, section)
                    if hasattr(section_config, field_name):
                        setattr(section_config, field_name, typed_value)

                        # Track metadata
                        metadata_key = f"{section}.{field_name}"
                        config._metadata[metadata_key] = ConfigMetadata(
                            source=ConfigSource.ENVIRONMENT, env_var=env_var
                        )

                        logger.debug(
                            f"Applied env override: {env_var} -> {section}.{field_name} = {typed_value}"
                        )

        return config

    def _apply_cli_overrides(
        self, config: UnifiedConfig, cli_overrides: Dict[str, Any]
    ) -> UnifiedConfig:
        """Apply CLI overrides with proper typing."""
        for key, value in cli_overrides.items():
            # Parse CLI key to config path (e.g., "evaluation.concurrency")
            path_parts = key.split(".")

            if len(path_parts) >= 2:
                section = path_parts[0]
                field_name = ".".join(path_parts[1:])

                if hasattr(config, section):
                    section_config = getattr(config, section)
                    if hasattr(section_config, field_name):
                        setattr(section_config, field_name, value)

                        # Track metadata
                        config._metadata[key] = ConfigMetadata(source=ConfigSource.CLI)

                        logger.debug(f"Applied CLI override: {key} = {value}")

        return config

    def _merge_config_data(
        self,
        config: UnifiedConfig,
        data: Dict[str, Any],
        source: ConfigSource,
        source_path: Optional[str] = None,
    ) -> UnifiedConfig:
        """Merge configuration data into config object."""

        for section_name, section_data in data.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section_config = getattr(config, section_name)

                for field_name, field_value in section_data.items():
                    if hasattr(section_config, field_name):
                        setattr(section_config, field_name, field_value)

                        # Track metadata
                        metadata_key = f"{section_name}.{field_name}"
                        config._metadata[metadata_key] = ConfigMetadata(
                            source=source, file_path=source_path
                        )

        return config

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass

        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass

        # JSON conversion for complex types
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass

        # Return as string
        return value

    def _validate_config(self, config: UnifiedConfig):
        """Validate the complete configuration."""
        errors = []

        # Validate each section
        for section_name in ["evaluation", "adapter", "dataset", "security", "observability"]:
            if hasattr(config, section_name):
                section_config = getattr(config, section_name)
                section_dict = self._dataclass_to_dict(section_config)
                section_errors = self.validator.validate_config(section_name, section_dict)
                errors.extend(section_errors)

        # Custom validation rules
        errors.extend(self._custom_validation(config))

        if errors:
            config._validation_errors = errors
            if config.environment == Environment.PRODUCTION:
                # Strict validation in production
                raise ConfigurationError(f"Configuration validation failed: {', '.join(errors)}")
            else:
                # Warnings in other environments
                for error in errors:
                    logger.warning(f"Configuration validation warning: {error}")

    def _custom_validation(self, config: UnifiedConfig) -> List[str]:
        """Apply custom validation rules."""
        errors = []

        # Validate resource constraints
        if config.evaluation.max_memory_gb and config.evaluation.max_memory_gb < 1.0:
            errors.append("evaluation.max_memory_gb should be at least 1.0 GB")

        # Validate security settings for production
        if config.environment == Environment.PRODUCTION:
            if not config.security.api_key_encryption:
                errors.append("security.api_key_encryption must be enabled in production")
            if not config.security.audit_log_path:
                errors.append("security.audit_log_path must be configured in production")

        # Validate observability settings
        if config.observability.tracing_enabled and not config.observability.tracing_endpoint:
            errors.append("observability.tracing_endpoint required when tracing is enabled")

        return errors

    def _dataclass_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert dataclass to dictionary for validation."""
        if is_dataclass(obj):
            return {field.name: getattr(obj, field.name) for field in fields(obj)}
        return {}

    def _is_cache_valid(self, config_file: Path) -> bool:
        """Check if cached configuration is still valid."""
        if not config_file.exists():
            return True  # No file to compare against

        file_path = str(config_file)
        if file_path not in self._file_watchers:
            return False

        current_mtime = config_file.stat().st_mtime
        cached_mtime = self._file_watchers[file_path]

        return current_mtime == cached_mtime

    def save_config(self, config: UnifiedConfig, output_path: Path, format: str = "yaml"):
        """Save configuration to file."""
        config_dict = {
            "environment": config.environment.value,
            "config_version": config.config_version,
            "evaluation": self._dataclass_to_dict(config.evaluation),
            "adapter": self._dataclass_to_dict(config.adapter),
            "dataset": self._dataclass_to_dict(config.dataset),
            "security": self._dataclass_to_dict(config.security),
            "observability": self._dataclass_to_dict(config.observability),
            "extensions": config.extensions,
        }

        # Remove None values and empty dicts
        config_dict = self._clean_dict(config_dict)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            if format.lower() in ["yaml", "yml"]:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                json.dump(config_dict, f, indent=2)
            else:
                raise ConfigurationError(f"Unsupported output format: {format}")

        logger.info(f"Configuration saved to: {output_path}")

    def _clean_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None values and empty containers from dictionary."""
        cleaned = {}
        for k, v in d.items():
            if isinstance(v, dict):
                nested = self._clean_dict(v)
                if nested:  # Only include non-empty dicts
                    cleaned[k] = nested
            elif isinstance(v, list):
                if v:  # Only include non-empty lists
                    cleaned[k] = v
            elif v is not None:  # Include all non-None values
                cleaned[k] = v
        return cleaned

    def get_config_hash(self, config: UnifiedConfig) -> str:
        """Generate hash for configuration state."""
        config_str = json.dumps(self._dataclass_to_dict(config), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def export_config_schema(self, output_path: Path):
        """Export JSON schema for configuration validation."""
        if not HAS_PYDANTIC:
            logger.warning("Pydantic not available, cannot export schema")
            return

        schemas = {}
        for section_name, schema_class in self.validator.validation_rules.items():
            if schema_class:
                schemas[section_name] = schema_class.schema()

        schema_doc = {
            "title": "OpenEval Configuration Schema",
            "type": "object",
            "properties": schemas,
        }

        with open(output_path, "w") as f:
            json.dump(schema_doc, f, indent=2)

        logger.info(f"Configuration schema exported to: {output_path}")


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(
    config_name: Optional[str] = None,
    config_path: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> UnifiedConfig:
    """Convenience function to load configuration."""
    manager = get_config_manager()
    return manager.load_config(config_name, config_path, cli_overrides)


def create_default_configs(config_dir: Path):
    """Create default configuration files for all environments."""
    manager = ConfigManager(config_dir)

    config_dir.mkdir(parents=True, exist_ok=True)

    for env in Environment:
        # Create environment-specific config
        config = UnifiedConfig(environment=env)

        # Adjust settings per environment
        if env == Environment.PRODUCTION:
            config.security.api_key_encryption = True
            config.security.audit_logging = True
            config.observability.tracing_enabled = True
            config.evaluation.log_level = "WARNING"
        elif env == Environment.DEVELOPMENT:
            config.evaluation.log_level = "DEBUG"
            config.evaluation.detailed_results = True

        config_file = config_dir / f"{env.value}.yaml"
        manager.save_config(config, config_file)

    # Create schema file
    schema_file = config_dir / "schema.json"
    manager.export_config_schema(schema_file)

    logger.info(f"Default configuration files created in: {config_dir}")


"""Configuration management and environment handling."""

from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConfigFormat(Enum):
    """Supported configuration formats."""

    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    ENV = "env"


@dataclass
class EvaluationSettings:
    """Evaluation-specific settings."""

    # Performance settings
    default_concurrency: int = 1
    default_max_retries: int = 3
    default_timeout: float = 30.0

    # Resource limits
    max_memory_mb: Optional[int] = None
    max_cpu_percent: Optional[float] = None
    max_disk_usage_mb: Optional[int] = None

    # Output settings
    default_output_format: str = "json"
    include_records_by_default: bool = False
    include_traces_by_default: bool = False

    # Cache settings
    default_cache_mode: str = "off"
    default_cache_ttl: Optional[float] = None
    cache_compression: bool = True

    # Error handling
    enable_robust_mode: bool = False
    max_retry_attempts: int = 3
    circuit_breaker_threshold: int = 5

    # Performance monitoring
    enable_benchmarking: bool = False
    enable_memory_profiling: bool = False
    save_performance_by_default: bool = False


@dataclass
class AdapterSettings:
    """Adapter-specific settings."""

    # API settings
    api_base_urls: Dict[str, str] = field(default_factory=dict)
    api_keys: Dict[str, str] = field(default_factory=dict)
    api_versions: Dict[str, str] = field(default_factory=dict)

    # Request settings
    default_temperature: float = 0.0
    default_max_tokens: int = 1024
    request_timeout: float = 30.0

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_factor: float = 2.0

    # Rate limiting
    rate_limit_rpm: Optional[int] = None
    rate_limit_tpm: Optional[int] = None
    burst_allowance: int = 5


@dataclass
class LoggingSettings:
    """Logging configuration."""

    level: str = "INFO"
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = False
    log_file_path: Optional[str] = None
    max_log_size_mb: int = 100
    log_retention_days: int = 30

    # Structured logging
    structured_logging: bool = False
    include_context: bool = True
    redact_sensitive: bool = True


@dataclass
class OpenEvalConfig:
    """Main OpenEval configuration."""

    # Core settings
    project_name: str = "openeval-project"
    version: str = "1.0.0"
    description: str = ""

    # Component settings
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    adapters: AdapterSettings = field(default_factory=AdapterSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    # Environment settings
    environment: str = "development"
    debug_mode: bool = False

    # Paths
    data_dir: str = "data"
    output_dir: str = "outputs"
    cache_dir: str = ".cache"
    log_dir: str = "logs"

    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenEvalConfig":
        """Create from dictionary."""
        # Handle nested dataclasses
        evaluation_data = data.get("evaluation", {})
        adapters_data = data.get("adapters", {})
        logging_data = data.get("logging", {})

        return cls(
            project_name=data.get("project_name", "openeval-project"),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            evaluation=EvaluationSettings(**evaluation_data),
            adapters=AdapterSettings(**adapters_data),
            logging=LoggingSettings(**logging_data),
            environment=data.get("environment", "development"),
            debug_mode=data.get("debug_mode", False),
            data_dir=data.get("data_dir", "data"),
            output_dir=data.get("output_dir", "outputs"),
            cache_dir=data.get("cache_dir", ".cache"),
            log_dir=data.get("log_dir", "logs"),
            custom=data.get("custom", {}),
        )


class ConfigManager:
    """Manages OpenEval configuration across different sources."""

    def __init__(self):
        self._config: Optional[OpenEvalConfig] = None
        self._config_sources: List[str] = []
        self._environment_overrides: Dict[str, Any] = {}

    def load_config(
        self,
        config_path: Optional[Union[str, Path]] = None,
        env_prefix: str = "OPENEVAL_",
        create_if_missing: bool = True,
    ) -> OpenEvalConfig:
        """Load configuration from various sources."""

        # Start with default config
        config_data = {}

        # 1. Load from config file if specified
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                config_data = self._load_config_file(config_path)
                self._config_sources.append(f"file:{config_path}")
            elif create_if_missing:
                self._create_default_config(config_path)
                config_data = self._load_config_file(config_path)
                self._config_sources.append(f"file:{config_path}")

        # 2. Look for config files in standard locations
        if not config_path:
            for location in self._get_config_search_paths():
                if location.exists():
                    config_data.update(self._load_config_file(location))
                    self._config_sources.append(f"file:{location}")
                    break

        # 3. Load environment variables
        env_overrides = self._load_environment_variables(env_prefix)
        if env_overrides:
            config_data = self._merge_configs(config_data, env_overrides)
            self._config_sources.append("environment")

        # 4. Create config object
        self._config = OpenEvalConfig.from_dict(config_data)

        return self._config

    def save_config(
        self,
        config: OpenEvalConfig,
        config_path: Union[str, Path],
        format: ConfigFormat = ConfigFormat.YAML,
    ):
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_data = config.to_dict()

        if format == ConfigFormat.JSON:
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

        elif format == ConfigFormat.YAML:
            with open(config_path, "w") as f:
                yaml.safe_dump(config_data, f, indent=2, default_flow_style=False)

        elif format == ConfigFormat.TOML:
            try:
                import tomli_w

                with open(config_path, "wb") as f:
                    tomli_w.dump(config_data, f)
            except ImportError:
                raise ImportError("tomli-w required for TOML support: pip install tomli-w")

        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_config(self) -> Optional[OpenEvalConfig]:
        """Get current config."""
        return self._config

    def get_config_sources(self) -> List[str]:
        """Get list of config sources used."""
        return self._config_sources.copy()

    def override_setting(self, key_path: str, value: Any):
        """Override a specific setting."""
        if not self._config:
            raise ValueError("No config loaded")

        # Parse key path (e.g., "evaluation.default_concurrency")
        keys = key_path.split(".")
        current = self._config

        # Navigate to parent
        for key in keys[:-1]:
            current = getattr(current, key)

        # Set final value
        setattr(current, keys[-1], value)

    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_path, "r") as f:
                if config_path.suffix.lower() in [".yml", ".yaml"]:
                    return yaml.safe_load(f) or {}

                elif config_path.suffix.lower() == ".json":
                    return json.load(f)

                elif config_path.suffix.lower() == ".toml":
                    try:
                        import tomllib

                        with open(config_path, "rb") as fb:
                            return tomllib.load(fb)
                    except ImportError:
                        raise ImportError("tomllib required for TOML support")

                else:
                    # Try YAML by default
                    return yaml.safe_load(f) or {}

        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}

    def _create_default_config(self, config_path: Path):
        """Create default configuration file."""
        default_config = OpenEvalConfig()
        self.save_config(default_config, config_path, ConfigFormat.YAML)
        logger.info(f"Created default config at {config_path}")

    def _get_config_search_paths(self) -> List[Path]:
        """Get standard config file search paths."""
        paths = []

        # Current directory
        for name in ["openeval.yaml", "openeval.yml", "openeval.json", ".openeval.yaml"]:
            paths.append(Path.cwd() / name)

        # Home directory
        home = Path.home()
        for name in [".openeval.yaml", ".openeval.yml", ".openeval.json"]:
            paths.append(home / name)

        # XDG config directory
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config:
            paths.append(Path(xdg_config) / "openeval" / "config.yaml")
        else:
            paths.append(home / ".config" / "openeval" / "config.yaml")

        return paths

    def _load_environment_variables(self, prefix: str) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}

        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix) :].lower()

                # Convert key path (e.g., evaluation_default_concurrency -> evaluation.default_concurrency)
                config_key = config_key.replace("_", ".")

                # Try to parse value as JSON first, then as string
                try:
                    parsed_value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    parsed_value = value

                # Set nested value
                self._set_nested_value(config, config_key, parsed_value)

        return config

    def _set_nested_value(self, config: Dict[str, Any], key_path: str, value: Any):
        """Set nested dictionary value using dot notation."""
        keys = key_path.split(".")
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result


class EnvironmentManager:
    """Manages environment-specific configurations and secrets."""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self._secrets: Dict[str, str] = {}

    def setup_environment(self, environment: str = "development"):
        """Setup environment-specific configuration."""
        config = self.config_manager.get_config()
        if not config:
            raise ValueError("No config loaded")

        # Update environment
        config.environment = environment

        # Environment-specific settings
        if environment == "development":
            config.debug_mode = True
            config.logging.level = "DEBUG"
            config.evaluation.enable_benchmarking = True

        elif environment == "testing":
            config.evaluation.default_concurrency = 1
            config.evaluation.enable_robust_mode = True
            config.logging.level = "INFO"

        elif environment == "production":
            config.debug_mode = False
            config.logging.level = "WARNING"
            config.evaluation.enable_robust_mode = True
            config.evaluation.save_performance_by_default = True

        # Load environment-specific secrets
        self._load_secrets(environment)

    def load_secrets(
        self,
        secrets_path: Optional[Union[str, Path]] = None,
        env_file: Optional[Union[str, Path]] = None,
    ):
        """Load secrets from various sources."""

        # Load from secrets file
        if secrets_path:
            secrets_path = Path(secrets_path)
            if secrets_path.exists():
                self._load_secrets_file(secrets_path)

        # Load from .env file
        if env_file:
            env_file = Path(env_file)
            if env_file.exists():
                self._load_env_file(env_file)

        # Auto-discover .env files
        for env_path in [Path(".env"), Path(".env.local")]:
            if env_path.exists():
                self._load_env_file(env_path)

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret value."""
        return self._secrets.get(key, default)

    def set_secret(self, key: str, value: str):
        """Set secret value."""
        self._secrets[key] = value

    def _load_secrets(self, environment: str):
        """Load environment-specific secrets."""
        # Look for secrets files
        secrets_files = [
            f".secrets.{environment}.yaml",
            f".secrets.{environment}.json",
            f"secrets/{environment}.yaml",
        ]

        for secrets_file in secrets_files:
            secrets_path = Path(secrets_file)
            if secrets_path.exists():
                self._load_secrets_file(secrets_path)
                break

    def _load_secrets_file(self, secrets_path: Path):
        """Load secrets from file."""
        try:
            with open(secrets_path, "r") as f:
                if secrets_path.suffix.lower() in [".yml", ".yaml"]:
                    secrets = yaml.safe_load(f) or {}
                else:
                    secrets = json.load(f)

                self._secrets.update(secrets)
                logger.info(f"Loaded secrets from {secrets_path}")

        except Exception as e:
            logger.warning(f"Failed to load secrets from {secrets_path}: {e}")

    def _load_env_file(self, env_path: Path):
        """Load environment variables from .env file."""
        try:
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if "=" in line:
                            key, value = line.split("=", 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            self._secrets[key] = value
                            os.environ[key] = value

            logger.info(f"Loaded environment from {env_path}")

        except Exception as e:
            logger.warning(f"Failed to load env file {env_path}: {e}")


def create_config_manager() -> ConfigManager:
    """Create and initialize config manager."""
    return ConfigManager()


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    environment: str = "development",
    load_secrets: bool = True,
) -> tuple[OpenEvalConfig, EnvironmentManager]:
    """Convenience function to load complete configuration."""

    # Create managers
    config_manager = create_config_manager()
    env_manager = EnvironmentManager(config_manager)

    # Load configuration
    config = config_manager.load_config(config_path)

    # Setup environment
    env_manager.setup_environment(environment)

    # Load secrets
    if load_secrets:
        env_manager.load_secrets()

    return config, env_manager


def create_default_config(output_path: Union[str, Path] = "openeval.yaml"):
    """Create a default configuration file."""
    config_manager = create_config_manager()
    default_config = OpenEvalConfig()
    config_manager.save_config(default_config, output_path, ConfigFormat.YAML)
    return output_path


def validate_config(config: OpenEvalConfig) -> List[str]:
    """Validate configuration and return list of issues."""
    issues = []

    # Validate evaluation settings
    if config.evaluation.default_concurrency < 1:
        issues.append("evaluation.default_concurrency must be >= 1")

    if config.evaluation.default_timeout <= 0:
        issues.append("evaluation.default_timeout must be > 0")

    if config.evaluation.max_retry_attempts < 0:
        issues.append("evaluation.max_retry_attempts must be >= 0")

    # Validate adapter settings
    if config.adapters.request_timeout <= 0:
        issues.append("adapters.request_timeout must be > 0")

    if config.adapters.max_retries < 0:
        issues.append("adapters.max_retries must be >= 0")

    # Validate logging settings
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if config.logging.level.upper() not in valid_levels:
        issues.append(f"logging.level must be one of: {', '.join(valid_levels)}")

    # Validate paths
    for path_name, path_value in [
        ("data_dir", config.data_dir),
        ("output_dir", config.output_dir),
        ("cache_dir", config.cache_dir),
        ("log_dir", config.log_dir),
    ]:
        if not path_value or not path_value.strip():
            issues.append(f"{path_name} cannot be empty")

    return issues


def validate_config_file(config_path: Union[str, Path]) -> tuple[bool, List[str]]:
    """Validate configuration file and return validation result."""
    config_path = Path(config_path)

    if not config_path.exists():
        return False, [f"Configuration file does not exist: {config_path}"]

    try:
        config_manager = create_config_manager()
        config = config_manager.load_config(config_path)
        issues = validate_config(config)

        if issues:
            return False, issues
        else:
            return True, []

    except Exception as e:
        return False, [f"Failed to load configuration: {e}"]


def validate_spec_against_config(spec_data: Dict[str, Any], config: OpenEvalConfig) -> List[str]:
    """Validate evaluation spec against configuration constraints."""
    issues = []

    # Check concurrency against config limits
    spec_concurrency = spec_data.get("concurrency", 1)
    if spec_concurrency > config.evaluation.default_concurrency * 2:
        issues.append(f"Spec concurrency ({spec_concurrency}) exceeds recommended limit")

    # Check timeout against config
    spec_timeout = spec_data.get("timeout")
    if spec_timeout and spec_timeout > config.evaluation.default_timeout * 2:
        issues.append(f"Spec timeout ({spec_timeout}s) exceeds recommended limit")

    # Check adapter settings
    adapter_config = spec_data.get("adapter", {})
    adapter_timeout = adapter_config.get("timeout")
    if adapter_timeout and adapter_timeout > config.adapters.request_timeout * 2:
        issues.append(f"Adapter timeout ({adapter_timeout}s) exceeds recommended limit")

    return issues
