"""Test observability configuration functionality."""

import pytest
from openeval.unified_config import ObservabilityConfig


class TestObservabilityConfig:
    """Test ObservabilityConfig dataclass."""

    def test_observability_config_default_values(self):
        """Test ObservabilityConfig with default values."""
        config = ObservabilityConfig()

        assert config.tracing_enabled is False
        assert config.tracing_endpoint is None
        assert config.metrics_enabled is True
        assert config.metrics_port == 8080
        assert config.trace_sample_rate == 0.1
        assert config.structured_logging is True
        assert config.log_correlation_id is True
        assert config.external_log_endpoint is None
        assert isinstance(config.custom_metrics, list)
        assert len(config.custom_metrics) == 0

    def test_observability_config_custom_values(self):
        """Test ObservabilityConfig with custom values."""
        config = ObservabilityConfig(
            tracing_enabled=True,
            tracing_endpoint="http://jaeger:14268",
            metrics_enabled=True,
            metrics_port=9091,
            trace_sample_rate=0.5,
            structured_logging=False,
            log_correlation_id=False,
        )

        assert config.tracing_enabled is True
        assert config.tracing_endpoint == "http://jaeger:14268"
        assert config.metrics_enabled is True
        assert config.metrics_port == 9091
        assert config.trace_sample_rate == 0.5
        assert config.structured_logging is False
        assert config.log_correlation_id is False

    def test_observability_config_with_unified_config(self):
        """Test ObservabilityConfig as part of unified configuration."""
        from openeval.unified_config import UnifiedConfig

        unified_config = UnifiedConfig()
        assert hasattr(unified_config, "observability")
        assert isinstance(unified_config.observability, ObservabilityConfig)

        unified_config.observability.tracing_enabled = True
        assert unified_config.observability.tracing_enabled is True

    def test_observability_config_serialization(self):
        """Test ObservabilityConfig serialization."""
        config = ObservabilityConfig(
            tracing_enabled=True, metrics_enabled=False, trace_sample_rate=0.8
        )

        config_dict = config.__dict__
        assert config_dict["tracing_enabled"] is True
        assert config_dict["metrics_enabled"] is False
        assert config_dict["trace_sample_rate"] == 0.8


class TestObservabilityConfigEnvironments:
    """Test observability config for different environments."""

    def test_development_observability_config(self):
        """Test observability config for development."""
        config = ObservabilityConfig(
            tracing_enabled=False, trace_sample_rate=1.0, log_correlation_id=True
        )

        assert config.tracing_enabled is False
        assert config.trace_sample_rate == 1.0
        assert config.log_correlation_id is True

    def test_production_observability_config(self):
        """Test observability config for production."""
        config = ObservabilityConfig(
            tracing_enabled=True,
            trace_sample_rate=0.1,
            structured_logging=True,
            external_log_endpoint="https://logs.production.com",
        )

        assert config.tracing_enabled is True
        assert config.trace_sample_rate == 0.1
        assert config.structured_logging is True
        assert config.external_log_endpoint == "https://logs.production.com"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
