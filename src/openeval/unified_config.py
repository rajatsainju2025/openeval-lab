"""Advanced unified configuration system with environment-aware configs and validation.

This module provides a comprehensive configuration management system that unifies
CLI flags, YAML configs, environment variables, and secrets into a single coherent
interface with validation, schema enforcement, and environment-specific overrides.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import time
import importlib.util
from .enhanced_logging import get_logger

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
