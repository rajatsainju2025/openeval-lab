"""Configuration presets for explainer workflows.

This module provides pre-built configurations for different
environments and use cases, making it easy to set up explainers
with sensible defaults.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .cache_manager import CacheManager, InMemoryCacheManager, NoOpCacheManager
from .rate_limiter import RateLimitConfig, RateLimitStrategy


class Environment(Enum):
    """Deployment environment types."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class CacheMode(Enum):
    """Cache mode settings."""

    DISABLED = "disabled"
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"


class LogLevel(Enum):
    """Logging level settings."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class CacheConfig:
    """Configuration for caching behavior."""

    mode: CacheMode = CacheMode.MEMORY
    ttl_seconds: int = 3600
    max_size: int = 1000
    namespace: str = "openeval"
    # Redis/Memcached settings
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0

    def create_cache_manager(self) -> CacheManager:
        """Create a cache manager based on config."""
        if self.mode == CacheMode.DISABLED:
            return NoOpCacheManager()
        elif self.mode == CacheMode.MEMORY:
            return InMemoryCacheManager()
        elif self.mode == CacheMode.REDIS:
            try:
                from .cache_backends import RedisCacheManager

                return RedisCacheManager(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                )
            except ImportError:
                # Fallback to memory cache if redis not available
                return InMemoryCacheManager()
        elif self.mode == CacheMode.MEMCACHED:
            try:
                from .cache_backends import MemcachedCacheManager

                return MemcachedCacheManager(
                    servers=[f"{self.host}:{self.port}"],
                )
            except ImportError:
                return InMemoryCacheManager()
        return NoOpCacheManager()


@dataclass
class RateLimitPreset:
    """Pre-configured rate limits for different tiers."""

    summary_limit: int = 100
    detailed_limit: int = 50
    expert_limit: int = 20
    window_seconds: float = 60.0
    global_limit: Optional[int] = None
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET

    def to_config(self) -> RateLimitConfig:
        """Convert to RateLimitConfig."""
        return RateLimitConfig(
            max_requests=self.global_limit or self.summary_limit,
            window_seconds=self.window_seconds,
            strategy=self.strategy,
        )


@dataclass
class RetryPreset:
    """Pre-configured retry settings."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging behavior."""

    level: LogLevel = LogLevel.INFO
    include_timestamps: bool = True
    include_metadata: bool = True
    log_requests: bool = True
    log_responses: bool = False  # Can be verbose
    log_errors: bool = True
    log_metrics: bool = False


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""

    enabled: bool = True
    collect_latency: bool = True
    collect_throughput: bool = True
    collect_cache_stats: bool = True
    collect_error_rates: bool = True
    export_format: str = "prometheus"  # prometheus, json, statsd


@dataclass
class ModelConfig:
    """Configuration for model selection."""

    default_model: str = "gpt-4"
    summary_model: str = "gpt-3.5-turbo"
    detailed_model: str = "gpt-4"
    expert_model: str = "gpt-4-turbo"
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout_seconds: float = 30.0


@dataclass
class ExplainerConfig:
    """Complete configuration for an explainer setup."""

    environment: Environment = Environment.DEVELOPMENT
    cache: CacheConfig = field(default_factory=CacheConfig)
    rate_limit: RateLimitPreset = field(default_factory=RateLimitPreset)
    retry: RetryPreset = field(default_factory=RetryPreset)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # Feature flags
    enable_caching: bool = True
    enable_rate_limiting: bool = True
    enable_retries: bool = True
    enable_events: bool = True
    enable_health_checks: bool = True

    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "environment": self.environment.value,
            "cache": {
                "mode": self.cache.mode.value,
                "ttl_seconds": self.cache.ttl_seconds,
                "max_size": self.cache.max_size,
            },
            "rate_limit": {
                "summary_limit": self.rate_limit.summary_limit,
                "detailed_limit": self.rate_limit.detailed_limit,
                "expert_limit": self.rate_limit.expert_limit,
                "window_seconds": self.rate_limit.window_seconds,
            },
            "retry": {
                "max_retries": self.retry.max_retries,
                "base_delay": self.retry.base_delay,
                "max_delay": self.retry.max_delay,
            },
            "logging": {
                "level": self.logging.level.value,
                "log_requests": self.logging.log_requests,
            },
            "model": {
                "default_model": self.model.default_model,
                "temperature": self.model.temperature,
            },
            "features": {
                "caching": self.enable_caching,
                "rate_limiting": self.enable_rate_limiting,
                "retries": self.enable_retries,
                "events": self.enable_events,
                "health_checks": self.enable_health_checks,
            },
            "custom": self.custom,
        }


# Pre-built configuration presets


def development_config() -> ExplainerConfig:
    """Configuration optimized for development.

    Features:
    - Memory caching with short TTL
    - Relaxed rate limits
    - Verbose logging
    - All features enabled for testing
    """
    return ExplainerConfig(
        environment=Environment.DEVELOPMENT,
        cache=CacheConfig(
            mode=CacheMode.MEMORY,
            ttl_seconds=300,  # 5 minutes
            max_size=500,
        ),
        rate_limit=RateLimitPreset(
            summary_limit=1000,
            detailed_limit=500,
            expert_limit=200,
            window_seconds=60.0,
        ),
        retry=RetryPreset(
            max_retries=2,
            base_delay=0.5,
            max_delay=10.0,
        ),
        logging=LoggingConfig(
            level=LogLevel.DEBUG,
            log_requests=True,
            log_responses=True,  # Verbose for debugging
            log_metrics=True,
        ),
        metrics=MetricsConfig(
            enabled=True,
            collect_latency=True,
            collect_throughput=True,
        ),
        model=ModelConfig(
            default_model="gpt-3.5-turbo",  # Cheaper for dev
            temperature=0.3,
            timeout_seconds=60.0,  # Longer timeout for debugging
        ),
        enable_caching=True,
        enable_rate_limiting=False,  # Disabled for easy dev
        enable_retries=True,
        enable_events=True,
        enable_health_checks=True,
    )


def testing_config() -> ExplainerConfig:
    """Configuration optimized for automated testing.

    Features:
    - No caching (predictable results)
    - No rate limiting
    - Minimal logging
    - Fast timeouts
    """
    return ExplainerConfig(
        environment=Environment.TESTING,
        cache=CacheConfig(
            mode=CacheMode.DISABLED,
        ),
        rate_limit=RateLimitPreset(
            summary_limit=10000,  # Effectively unlimited
            detailed_limit=10000,
            expert_limit=10000,
        ),
        retry=RetryPreset(
            max_retries=1,
            base_delay=0.1,
            max_delay=1.0,
        ),
        logging=LoggingConfig(
            level=LogLevel.ERROR,  # Only errors
            log_requests=False,
            log_responses=False,
            log_metrics=False,
        ),
        metrics=MetricsConfig(
            enabled=False,  # Disabled for speed
        ),
        model=ModelConfig(
            default_model="gpt-3.5-turbo",
            temperature=0.0,  # Deterministic
            timeout_seconds=10.0,  # Fast fail
        ),
        enable_caching=False,
        enable_rate_limiting=False,
        enable_retries=False,  # Fast fail for tests
        enable_events=False,
        enable_health_checks=False,
    )


def staging_config() -> ExplainerConfig:
    """Configuration for staging/pre-production.

    Features:
    - Production-like settings
    - More verbose logging
    - Relaxed rate limits
    """
    return ExplainerConfig(
        environment=Environment.STAGING,
        cache=CacheConfig(
            mode=CacheMode.MEMORY,
            ttl_seconds=1800,  # 30 minutes
            max_size=2000,
        ),
        rate_limit=RateLimitPreset(
            summary_limit=500,
            detailed_limit=250,
            expert_limit=100,
            window_seconds=60.0,
        ),
        retry=RetryPreset(
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
        ),
        logging=LoggingConfig(
            level=LogLevel.INFO,
            log_requests=True,
            log_responses=False,
            log_metrics=True,
        ),
        metrics=MetricsConfig(
            enabled=True,
            collect_latency=True,
            collect_throughput=True,
            collect_cache_stats=True,
            collect_error_rates=True,
        ),
        model=ModelConfig(
            default_model="gpt-4",
            temperature=0.3,
            timeout_seconds=30.0,
        ),
        enable_caching=True,
        enable_rate_limiting=True,
        enable_retries=True,
        enable_events=True,
        enable_health_checks=True,
    )


def production_config() -> ExplainerConfig:
    """Configuration optimized for production.

    Features:
    - Redis caching for distributed systems
    - Strict rate limits
    - Error-only logging
    - Full metrics collection
    """
    return ExplainerConfig(
        environment=Environment.PRODUCTION,
        cache=CacheConfig(
            mode=CacheMode.REDIS,
            ttl_seconds=3600,  # 1 hour
            max_size=10000,
            host="localhost",
            port=6379,
        ),
        rate_limit=RateLimitPreset(
            summary_limit=100,
            detailed_limit=50,
            expert_limit=20,
            window_seconds=60.0,
            global_limit=200,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
        ),
        retry=RetryPreset(
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0,
            jitter=True,
        ),
        logging=LoggingConfig(
            level=LogLevel.WARNING,
            log_requests=False,  # Too verbose
            log_responses=False,
            log_errors=True,
            log_metrics=False,  # Use metrics system instead
        ),
        metrics=MetricsConfig(
            enabled=True,
            collect_latency=True,
            collect_throughput=True,
            collect_cache_stats=True,
            collect_error_rates=True,
            export_format="prometheus",
        ),
        model=ModelConfig(
            default_model="gpt-4",
            summary_model="gpt-3.5-turbo",
            detailed_model="gpt-4",
            expert_model="gpt-4-turbo",
            temperature=0.3,
            max_tokens=2048,
            timeout_seconds=30.0,
        ),
        enable_caching=True,
        enable_rate_limiting=True,
        enable_retries=True,
        enable_events=True,
        enable_health_checks=True,
    )


def high_throughput_config() -> ExplainerConfig:
    """Configuration for high-throughput scenarios.

    Features:
    - Aggressive caching
    - Higher rate limits
    - Faster, cheaper models
    """
    return ExplainerConfig(
        environment=Environment.PRODUCTION,
        cache=CacheConfig(
            mode=CacheMode.REDIS,
            ttl_seconds=7200,  # 2 hours
            max_size=50000,
        ),
        rate_limit=RateLimitPreset(
            summary_limit=500,
            detailed_limit=250,
            expert_limit=100,
            window_seconds=60.0,
            global_limit=1000,
            strategy=RateLimitStrategy.TOKEN_BUCKET,
        ),
        retry=RetryPreset(
            max_retries=2,
            base_delay=0.5,
            max_delay=10.0,
        ),
        logging=LoggingConfig(
            level=LogLevel.ERROR,
            log_requests=False,
            log_responses=False,
        ),
        metrics=MetricsConfig(
            enabled=True,
            export_format="statsd",
        ),
        model=ModelConfig(
            default_model="gpt-3.5-turbo",
            summary_model="gpt-3.5-turbo",
            detailed_model="gpt-3.5-turbo",
            expert_model="gpt-4",
            temperature=0.2,
            max_tokens=1024,
            timeout_seconds=15.0,
        ),
        enable_caching=True,
        enable_rate_limiting=True,
        enable_retries=True,
        enable_events=False,  # Reduce overhead
        enable_health_checks=True,
    )


def low_latency_config() -> ExplainerConfig:
    """Configuration for low-latency requirements.

    Features:
    - Memory caching for speed
    - Shorter timeouts
    - Minimal middleware
    """
    return ExplainerConfig(
        environment=Environment.PRODUCTION,
        cache=CacheConfig(
            mode=CacheMode.MEMORY,
            ttl_seconds=1800,
            max_size=5000,
        ),
        rate_limit=RateLimitPreset(
            summary_limit=200,
            detailed_limit=100,
            expert_limit=50,
            window_seconds=60.0,
        ),
        retry=RetryPreset(
            max_retries=1,  # Fast fail
            base_delay=0.2,
            max_delay=2.0,
        ),
        logging=LoggingConfig(
            level=LogLevel.ERROR,
            log_requests=False,
            log_responses=False,
            log_metrics=False,
        ),
        metrics=MetricsConfig(
            enabled=True,
            collect_latency=True,
            collect_throughput=False,
            collect_cache_stats=False,
        ),
        model=ModelConfig(
            default_model="gpt-3.5-turbo",
            temperature=0.2,
            max_tokens=512,  # Shorter responses
            timeout_seconds=10.0,  # Fast timeout
        ),
        enable_caching=True,
        enable_rate_limiting=True,
        enable_retries=False,  # No retry delay
        enable_events=False,
        enable_health_checks=False,
    )


def budget_config() -> ExplainerConfig:
    """Configuration for cost-conscious usage.

    Features:
    - Aggressive caching
    - Cheaper models
    - Limited rate
    """
    return ExplainerConfig(
        environment=Environment.PRODUCTION,
        cache=CacheConfig(
            mode=CacheMode.MEMORY,
            ttl_seconds=86400,  # 24 hours
            max_size=10000,
        ),
        rate_limit=RateLimitPreset(
            summary_limit=50,
            detailed_limit=20,
            expert_limit=5,
            window_seconds=60.0,
            global_limit=100,
        ),
        retry=RetryPreset(
            max_retries=2,
            base_delay=2.0,
            max_delay=30.0,
        ),
        logging=LoggingConfig(
            level=LogLevel.WARNING,
        ),
        metrics=MetricsConfig(
            enabled=True,
            collect_latency=False,
            collect_throughput=True,  # Track usage
        ),
        model=ModelConfig(
            default_model="gpt-3.5-turbo",
            summary_model="gpt-3.5-turbo",
            detailed_model="gpt-3.5-turbo",
            expert_model="gpt-4",  # Only expert uses gpt-4
            temperature=0.3,
            max_tokens=1024,
        ),
        enable_caching=True,
        enable_rate_limiting=True,
        enable_retries=True,
        enable_events=False,
        enable_health_checks=False,
    )


# Registry of preset configurations
_PRESET_REGISTRY: Dict[str, ExplainerConfig] = {}


def register_preset(name: str, config: ExplainerConfig) -> None:
    """Register a custom configuration preset.

    Args:
        name: Name for the preset.
        config: Configuration to register.
    """
    _PRESET_REGISTRY[name] = config


def get_preset(name: str) -> Optional[ExplainerConfig]:
    """Get a registered preset by name.

    Args:
        name: Preset name.

    Returns:
        Configuration or None if not found.
    """
    # Check custom registry first
    if name in _PRESET_REGISTRY:
        return _PRESET_REGISTRY[name]

    # Built-in presets
    presets = {
        "development": development_config,
        "dev": development_config,
        "testing": testing_config,
        "test": testing_config,
        "staging": staging_config,
        "production": production_config,
        "prod": production_config,
        "high-throughput": high_throughput_config,
        "low-latency": low_latency_config,
        "budget": budget_config,
    }

    factory = presets.get(name.lower())
    return factory() if factory else None


def list_presets() -> List[str]:
    """List all available preset names.

    Returns:
        List of preset names.
    """
    builtin = [
        "development",
        "testing",
        "staging",
        "production",
        "high-throughput",
        "low-latency",
        "budget",
    ]
    custom = list(_PRESET_REGISTRY.keys())
    return builtin + custom


def get_config_for_environment(env: str) -> ExplainerConfig:
    """Get the appropriate configuration for an environment.

    Args:
        env: Environment name (development, testing, staging, production).

    Returns:
        Appropriate ExplainerConfig.
    """
    env = env.lower()
    if env in ("development", "dev", "local"):
        return development_config()
    elif env in ("testing", "test", "ci"):
        return testing_config()
    elif env in ("staging", "stage", "preprod"):
        return staging_config()
    elif env in ("production", "prod", "live"):
        return production_config()
    else:
        # Default to development for unknown environments
        return development_config()


def create_config(
    environment: str = "development",
    *,
    cache_mode: Optional[str] = None,
    cache_ttl: Optional[int] = None,
    rate_limit: Optional[int] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> ExplainerConfig:
    """Create a configuration with customizations.

    Starts with an environment preset and applies custom overrides.

    Args:
        environment: Base environment preset to use.
        cache_mode: Override cache mode (disabled, memory, redis).
        cache_ttl: Override cache TTL in seconds.
        rate_limit: Override summary rate limit.
        model: Override default model.
        **kwargs: Additional custom settings.

    Returns:
        Customized ExplainerConfig.
    """
    config = get_config_for_environment(environment)

    if cache_mode:
        try:
            config.cache.mode = CacheMode(cache_mode)
        except ValueError:
            pass

    if cache_ttl is not None:
        config.cache.ttl_seconds = cache_ttl

    if rate_limit is not None:
        config.rate_limit.summary_limit = rate_limit

    if model:
        config.model.default_model = model

    config.custom.update(kwargs)

    return config
