"""Configuration management and environment handling."""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import tempfile
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
    def from_dict(cls, data: Dict[str, Any]) -> 'OpenEvalConfig':
        """Create from dictionary."""
        # Handle nested dataclasses
        evaluation_data = data.get('evaluation', {})
        adapters_data = data.get('adapters', {})
        logging_data = data.get('logging', {})
        
        return cls(
            project_name=data.get('project_name', 'openeval-project'),
            version=data.get('version', '1.0.0'),
            description=data.get('description', ''),
            evaluation=EvaluationSettings(**evaluation_data),
            adapters=AdapterSettings(**adapters_data),
            logging=LoggingSettings(**logging_data),
            environment=data.get('environment', 'development'),
            debug_mode=data.get('debug_mode', False),
            data_dir=data.get('data_dir', 'data'),
            output_dir=data.get('output_dir', 'outputs'),
            cache_dir=data.get('cache_dir', '.cache'),
            log_dir=data.get('log_dir', 'logs'),
            custom=data.get('custom', {})
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
        create_if_missing: bool = True
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
        format: ConfigFormat = ConfigFormat.YAML
    ):
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = config.to_dict()
        
        if format == ConfigFormat.JSON:
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
        
        elif format == ConfigFormat.YAML:
            with open(config_path, 'w') as f:
                yaml.safe_dump(config_data, f, indent=2, default_flow_style=False)
        
        elif format == ConfigFormat.TOML:
            try:
                import tomli_w
                with open(config_path, 'wb') as f:
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
        keys = key_path.split('.')
        current = self._config
        
        # Navigate to parent
        for key in keys[:-1]:
            current = getattr(current, key)
        
        # Set final value
        setattr(current, keys[-1], value)
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    return yaml.safe_load(f) or {}
                
                elif config_path.suffix.lower() == '.json':
                    return json.load(f)
                
                elif config_path.suffix.lower() == '.toml':
                    try:
                        import tomllib
                        with open(config_path, 'rb') as fb:
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
        for name in ['openeval.yaml', 'openeval.yml', 'openeval.json', '.openeval.yaml']:
            paths.append(Path.cwd() / name)
        
        # Home directory
        home = Path.home()
        for name in ['.openeval.yaml', '.openeval.yml', '.openeval.json']:
            paths.append(home / name)
        
        # XDG config directory
        xdg_config = os.environ.get('XDG_CONFIG_HOME')
        if xdg_config:
            paths.append(Path(xdg_config) / 'openeval' / 'config.yaml')
        else:
            paths.append(home / '.config' / 'openeval' / 'config.yaml')
        
        return paths
    
    def _load_environment_variables(self, prefix: str) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()
                
                # Convert key path (e.g., evaluation_default_concurrency -> evaluation.default_concurrency)
                config_key = config_key.replace('_', '.')
                
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
        keys = key_path.split('.')
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
        env_file: Optional[Union[str, Path]] = None
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
        for env_path in [Path('.env'), Path('.env.local')]:
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
            f"secrets/{environment}.yaml"
        ]
        
        for secrets_file in secrets_files:
            secrets_path = Path(secrets_file)
            if secrets_path.exists():
                self._load_secrets_file(secrets_path)
                break
    
    def _load_secrets_file(self, secrets_path: Path):
        """Load secrets from file."""
        try:
            with open(secrets_path, 'r') as f:
                if secrets_path.suffix.lower() in ['.yml', '.yaml']:
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
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
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
    load_secrets: bool = True
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
        ("log_dir", config.log_dir)
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
