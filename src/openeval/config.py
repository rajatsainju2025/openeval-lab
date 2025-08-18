"""Configuration management system for OpenEval Lab."""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field


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
        Path.home() / ".openeval" / "config.yaml"
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
        with open(path, 'w') as f:
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
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif path.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        return self._dict_to_config(data)
    
    def _dict_to_config(self, data: Dict[str, Any]) -> OpenEvalConfig:
        """Convert dictionary to OpenEvalConfig object."""
        
        # Extract sub-configurations
        eval_config = None
        if 'evaluation' in data:
            eval_config = EvaluationConfig(**data['evaluation'])
        
        web_config = None
        if 'web' in data:
            web_config = WebConfig(**data['web'])
        
        # Create main config
        main_data = {k: v for k, v in data.items() 
                    if k not in ['evaluation', 'web']}
        
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
        parts = key.split('.')
        
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
