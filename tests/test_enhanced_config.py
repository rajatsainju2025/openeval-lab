"""Tests for enhanced configuration system."""

import pytest
import json
import tempfile
from pathlib import Path
from openeval.config import (
    ConfigTemplate,
    ConfigProfile,
    JSONConfigLoader,
    YAMLConfigLoader,
    TemplateEngine,
    EnhancedConfigManager,
    create_base_template,
    create_development_profile,
    create_production_profile,
    HAS_JINJA2,
)


class TestConfigTemplate:
    """Test ConfigTemplate class."""

    def test_init(self):
        """Test initialization."""
        template = ConfigTemplate(
            name="test",
            description="Test template",
            extends="base",
            config={"key": "value"},
            variables={"var": "value"},
        )

        assert template.name == "test"
        assert template.description == "Test template"
        assert template.extends == "base"
        assert template.config == {"key": "value"}
        assert template.variables == {"var": "value"}


class TestConfigProfile:
    """Test ConfigProfile class."""

    def test_init(self):
        """Test initialization."""
        profile = ConfigProfile(
            name="test",
            templates=["base"],
            overrides={"key": "override"},
            variables={"var": "value"},
            environment="test",
        )

        assert profile.name == "test"
        assert profile.templates == ["base"]
        assert profile.overrides == {"key": "override"}
        assert profile.variables == {"var": "value"}
        assert profile.environment == "test"


class TestJSONConfigLoader:
    """Test JSON configuration loader."""

    def test_load_from_dict(self):
        """Test loading from dictionary."""
        loader = JSONConfigLoader()
        data = {"key": "value"}
        result = loader.load(data)
        assert result == data

    def test_load_from_file(self):
        """Test loading from JSON file."""
        loader = JSONConfigLoader()
        data = {"key": "value", "number": 42}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            result = loader.load(temp_path)
            assert result == data
        finally:
            Path(temp_path).unlink()

    def test_save_to_file(self):
        """Test saving to JSON file."""
        loader = JSONConfigLoader()
        data = {"key": "value"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            loader.save(data, temp_path)
            with open(temp_path, "r") as f:
                loaded = json.load(f)
            assert loaded == data
        finally:
            Path(temp_path).unlink()


class TestYAMLConfigLoader:
    """Test YAML configuration loader."""

    def test_load_from_dict(self):
        """Test loading from dictionary."""
        loader = YAMLConfigLoader()
        data = {"key": "value"}
        result = loader.load(data)
        assert result == data

    def test_load_from_file(self):
        """Test loading from YAML file."""
        loader = YAMLConfigLoader()
        data = {"key": "value", "nested": {"item": 42}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value\nnested:\n  item: 42\n")
            temp_path = f.name

        try:
            result = loader.load(temp_path)
            assert result == data
        finally:
            Path(temp_path).unlink()


class TestTemplateEngine:
    """Test template engine."""

    def test_render_template_simple(self):
        """Test simple template rendering."""
        engine = TemplateEngine()
        template = "Hello {{ name }}!"
        variables = {"name": "World"}
        result = engine.render_template(template, variables)
        assert result == "Hello World!"

    def test_render_config(self):
        """Test configuration rendering."""
        engine = TemplateEngine()
        config = {
            "database": {"url": "{{ db_url }}", "port": "{{ db_port }}"},
            "logging": {"level": "{{ log_level }}"},
        }
        variables = {"db_url": "localhost", "db_port": "5432", "log_level": "DEBUG"}

        result = engine.render_config(config, variables)
        expected = {"database": {"url": "localhost", "port": "5432"}, "logging": {"level": "DEBUG"}}
        assert result == expected

    @pytest.mark.skipif(not HAS_JINJA2, reason="Jinja2 not available")
    def test_jinja2_rendering(self):
        """Test Jinja2 template rendering."""
        engine = TemplateEngine()
        template = "Value: {{ value | upper }}"
        variables = {"value": "test"}
        result = engine.render_template(template, variables)
        assert result == "Value: TEST"


class TestEnhancedConfigManager:
    """Test enhanced configuration manager."""

    def test_register_template(self):
        """Test registering templates."""
        manager = EnhancedConfigManager()
        template = ConfigTemplate(name="test", description="Test template")

        manager.register_template(template)
        assert "test" in manager.templates

    def test_register_profile(self):
        """Test registering profiles."""
        manager = EnhancedConfigManager()
        profile = ConfigProfile(name="test", templates=["base"])

        manager.register_profile(profile)
        assert "test" in manager.profiles

    def test_build_config_simple(self):
        """Test building simple configuration."""
        manager = EnhancedConfigManager()

        # Register template
        template = ConfigTemplate(
            name="base", description="Base template", config={"key": "value", "number": 42}
        )
        manager.register_template(template)

        # Register profile
        profile = ConfigProfile(name="test", templates=["base"], overrides={"key": "override"})
        manager.register_profile(profile)

        result = manager.build_config("test")
        expected = {"key": "override", "number": 42}
        assert result == expected

    def test_template_inheritance(self):
        """Test template inheritance."""
        manager = EnhancedConfigManager()

        # Base template
        base_template = ConfigTemplate(
            name="base",
            description="Base template",
            config={"shared": "base_value", "base_only": "base"},
        )
        manager.register_template(base_template)

        # Extended template
        extended_template = ConfigTemplate(
            name="extended",
            description="Extended template",
            extends="base",
            config={"shared": "extended_value", "extended_only": "extended"},
        )
        manager.register_template(extended_template)

        # Profile using extended template
        profile = ConfigProfile(name="test", templates=["extended"])
        manager.register_profile(profile)

        result = manager.build_config("test")
        expected = {"shared": "extended_value", "base_only": "base", "extended_only": "extended"}
        assert result == expected

    def test_variable_rendering(self):
        """Test variable rendering in configuration."""
        manager = EnhancedConfigManager()

        template = ConfigTemplate(
            name="templated",
            description="Templated config",
            config={
                "database": {"url": "{{ db_host }}:{{ db_port }}", "name": "{{ db_name }}"},
                "logging": {"file": "{{ log_file }}"},
            },
            variables={"db_host": "localhost", "db_port": "5432", "db_name": "testdb"},
        )
        manager.register_template(template)

        profile = ConfigProfile(
            name="test", templates=["templated"], variables={"log_file": "/var/log/app.log"}
        )
        manager.register_profile(profile)

        result = manager.build_config("test")
        expected = {
            "database": {"url": "localhost:5432", "name": "testdb"},
            "logging": {"file": "/var/log/app.log"},
        }
        assert result == expected

    def test_environment_variables(self):
        """Test environment variable integration."""
        manager = EnhancedConfigManager()

        template = ConfigTemplate(
            name="env_test", description="Environment test", config={"api_key": "{{ api_key }}"}
        )
        manager.register_template(template)

        profile = ConfigProfile(name="test", templates=["env_test"])
        manager.register_profile(profile)

        # Set environment variable
        import os

        os.environ["OPENEVAL_API_KEY"] = "secret123"

        try:
            result = manager.build_config("test")
            assert result["api_key"] == "secret123"
        finally:
            del os.environ["OPENEVAL_API_KEY"]

    def test_config_validation(self):
        """Test configuration validation."""
        manager = EnhancedConfigManager()

        # Valid config
        valid_config = {
            "logging": {"level": "INFO"},
            "evaluation": {"timeout": 300},
            "storage": {"type": "local", "path": "/data"},
        }
        errors = manager.validate_config(valid_config)
        assert len(errors) == 0

        # Invalid config
        invalid_config = {
            "logging": {"level": "INVALID"},
            "storage": {"type": "local"},
            # Missing evaluation section
        }
        errors = manager.validate_config(invalid_config)
        assert len(errors) > 0
        assert any("Missing required section: evaluation" in error for error in errors)
        assert any("Invalid log level" in error for error in errors)

    def test_unknown_profile(self):
        """Test error when profile not found."""
        manager = EnhancedConfigManager()
        with pytest.raises(ValueError, match="Profile 'unknown' not found"):
            manager.build_config("unknown")

    def test_unknown_template(self):
        """Test error when template not found."""
        manager = EnhancedConfigManager()

        profile = ConfigProfile(name="test", templates=["unknown"])
        manager.register_profile(profile)

        with pytest.raises(ValueError, match="Template 'unknown' not found"):
            manager.build_config("test")


class TestPredefinedTemplates:
    """Test predefined templates and profiles."""

    def test_create_base_template(self):
        """Test creating base template."""
        template = create_base_template()
        assert template.name == "base"
        assert "logging" in template.config
        assert "evaluation" in template.config
        assert "storage" in template.config
        assert "log_level" in template.variables

    def test_create_development_profile(self):
        """Test creating development profile."""
        profile = create_development_profile()
        assert profile.name == "development"
        assert "base" in profile.templates
        assert profile.environment == "development"
        assert "log_level" in profile.variables

    def test_create_production_profile(self):
        """Test creating production profile."""
        profile = create_production_profile()
        assert profile.name == "production"
        assert profile.environment == "production"
        assert "s3_bucket" in profile.variables
