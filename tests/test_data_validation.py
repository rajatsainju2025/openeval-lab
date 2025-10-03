"""Tests for data validation system."""

import pytest
from unittest.mock import patch
from openeval.data_validation import (
    DataValidator,
    ValidationRule,
    PydanticDataValidator,
    create_standard_validator,
    HAS_PYDANTIC,
    HAS_JSONSCHEMA,
)


class TestDataValidator:
    """Test the DataValidator class."""

    def test_add_rule(self):
        """Test adding validation rules."""
        validator = DataValidator()
        rule = ValidationRule(name="test_rule", description="Test rule", validator=lambda x: True)
        validator.add_rule(rule)
        assert "test_rule" in validator.rules

    def test_validate_data_with_rules(self):
        """Test validating data with rules."""
        validator = DataValidator()

        # Add a passing rule
        validator.add_rule(
            ValidationRule(
                name="always_pass", description="Always passes", validator=lambda x: True
            )
        )

        # Add a failing rule
        validator.add_rule(
            ValidationRule(
                name="always_fail", description="Always fails", validator=lambda x: False
            )
        )

        result = validator.validate_data("test_data")
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "always_fail" in result.errors[0]

    def test_validate_data_with_warnings(self):
        """Test validation with warning rules."""
        validator = DataValidator()

        validator.add_rule(
            ValidationRule(
                name="warning_rule",
                description="Warning rule",
                validator=lambda x: False,
                severity="warning",
            )
        )

        result = validator.validate_data("test_data")
        assert result.is_valid  # Warnings don't make it invalid
        assert len(result.warnings) == 1

    def test_validate_json_schema_without_jsonschema(self):
        """Test JSON schema validation when jsonschema is not available."""
        with patch("openeval.data_validation.HAS_JSONSCHEMA", False):
            validator = DataValidator()
            result = validator.validate_json_schema({}, "test_schema")
            assert not result.is_valid
            assert "not available" in result.errors[0]

    @pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not available")
    def test_validate_json_schema_with_jsonschema(self):
        """Test JSON schema validation when jsonschema is available."""
        validator = DataValidator()

        # Add a simple schema
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        validator.add_schema("test_schema", schema)

        # Valid data
        result = validator.validate_json_schema({"name": "test"}, "test_schema")
        assert result.is_valid

        # Invalid data
        result = validator.validate_json_schema({}, "test_schema")
        assert not result.is_valid

    def test_unknown_schema(self):
        """Test validation with unknown schema."""
        validator = DataValidator()
        result = validator.validate_json_schema({}, "unknown_schema")
        assert not result.is_valid
        assert "not found" in result.errors[0]


class TestPydanticDataValidator:
    """Test the PydanticDataValidator class."""

    @pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not available")
    def test_validate_model_success(self):
        """Test successful model validation."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            value: int = 0

        validator = PydanticDataValidator()
        result = validator.validate_model(TestModel, {"name": "test", "value": 42})
        assert result.is_valid
        assert result.metadata["model"] == "TestModel"

    @pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not available")
    def test_validate_model_failure(self):
        """Test failed model validation."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            value: int

        validator = PydanticDataValidator()
        result = validator.validate_model(TestModel, {"name": "test", "value": "not_a_number"})
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_pydantic_not_available(self):
        """Test behavior when pydantic is not available."""
        with patch("openeval.data_validation.HAS_PYDANTIC", False):
            with pytest.raises(ImportError):
                PydanticDataValidator()


class TestStandardValidator:
    """Test the standard validator factory."""

    def test_create_standard_validator(self):
        """Test creating a standard validator."""
        validator = create_standard_validator()
        assert len(validator.rules) > 0
        assert "not_empty" in validator.rules
        assert "valid_json" in validator.rules

    def test_standard_rules(self):
        """Test standard validation rules."""
        validator = create_standard_validator()

        # Test not_empty rule
        result = validator.validate_data("", ["not_empty"])
        assert not result.is_valid

        result = validator.validate_data("test", ["not_empty"])
        assert result.is_valid

        # Test valid_json rule
        result = validator.validate_data('{"key": "value"}', ["valid_json"])
        assert result.is_valid

        result = validator.validate_data("invalid json", ["valid_json"])
        assert not result.is_valid
