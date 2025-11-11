"""
Configuration precedence and consolidation utilities.

Provides unified configuration handling with clear precedence order:
1. Environment variables (highest priority)
2. CLI arguments
3. Config file (YAML/JSON)
4. Defaults (lowest priority)
"""

import os
from typing import Any, Dict, Optional


class ConfigPrecedence:
    """Handle configuration with clear precedence order."""

    @staticmethod
    def resolve(
        env_var: Optional[str] = None,
        cli_arg: Any = None,
        config_file_value: Any = None,
        default: Any = None,
    ) -> Any:
        """Resolve configuration value with precedence.

        Order (highest to lowest):
        1. Environment variable
        2. CLI argument (if not default/None)
        3. Config file value
        4. Default value

        Args:
            env_var: Environment variable name to check
            cli_arg: CLI argument value
            config_file_value: Value from config file
            default: Default value

        Returns:
            Resolved configuration value
        """
        # Check environment variable first
        if env_var:
            env_value = os.environ.get(env_var.upper())
            if env_value is not None:
                return ConfigPrecedence._parse_value(env_value)

        # Check CLI argument
        if cli_arg is not None and cli_arg != default:
            return cli_arg

        # Check config file
        if config_file_value is not None:
            return config_file_value

        # Use default
        return default

    @staticmethod
    def _parse_value(value: str) -> Any:
        """Parse environment variable value.

        Args:
            value: String value from environment

        Returns:
            Parsed value (bool, int, float, or string)
        """
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value


class ConsolidatedConfig:
    """Unified configuration from multiple sources."""

    def __init__(self):
        """Initialize consolidated configuration."""
        self._config: Dict[str, Any] = {}
        self._sources: Dict[str, str] = {}  # Track source of each config

    def set(
        self,
        key: str,
        value: Any,
        source: str = "default",
    ) -> None:
        """Set configuration value with source tracking.

        Args:
            key: Configuration key
            value: Configuration value
            source: Where this came from (env, cli, file, default)
        """
        self._config[key] = value
        self._sources[key] = source

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key
            default: Default if not found

        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)

    def get_with_source(self, key: str, default: Any = None) -> tuple[Any, str]:
        """Get configuration value with its source.

        Args:
            key: Configuration key
            default: Default if not found

        Returns:
            Tuple of (value, source)
        """
        value = self._config.get(key, default)
        source = self._sources.get(key, "unknown")
        return value, source

    def to_dict(self) -> Dict[str, Any]:
        """Get all configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return self._config.copy()

    def debug_info(self) -> str:
        """Get debug information about configuration sources.

        Returns:
            Formatted debug string
        """
        lines = ["Configuration Sources:\n"]
        for key, value in sorted(self._config.items()):
            source = self._sources.get(key, "unknown")
            lines.append(f"  {key}: {value} (source: {source})")
        return "\n".join(lines)


# Common configuration templates
DEFAULT_CACHE_CONFIG = {
    "enabled": True,
    "backend": "sqlite",
    "compression": "zlib",
    "memory_cache_size": 1000,
    "ttl": 86400,
    "bloom_filter_size": 100000,
}

DEFAULT_EVALUATION_CONFIG = {
    "batch_size": 32,
    "timeout": 30.0,
    "retry_count": 3,
    "stream": False,
    "verbose": False,
}

DEFAULT_RESOURCE_CONFIG = {
    "monitor": True,
    "memory_threshold_percent": 85.0,
    "cpu_threshold_percent": 90.0,
    "check_interval": 5.0,
}
