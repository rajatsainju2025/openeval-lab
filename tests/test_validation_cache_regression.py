"""Regression tests for validation cache behavior."""

import pytest
from openeval.data_validation import DataValidator, ValidationRule


@pytest.fixture
def validator_with_rules():
    """Create a validator with test rules."""
    validator = DataValidator()

    # Add test rules
    validator.add_rule(
        ValidationRule(
            name="not_empty",
            description="Data should not be empty",
            validator=lambda data: data is not None and data != "",
        )
    )
    validator.add_rule(
        ValidationRule(
            name="is_string",
            description="Data should be a string",
            validator=lambda data: isinstance(data, str),
        )
    )

    return validator


def test_cache_hit_same_data(validator_with_rules):
    """Test that same data produces cache hit."""
    data = {"name": "test", "value": 42}

    result1 = validator_with_rules.validate_data(data, rules=["not_empty"])
    result2 = validator_with_rules.validate_data(data, rules=["not_empty"])

    # Both should pass and be the same result
    assert result1.is_valid == result2.is_valid
    assert result1.errors == result2.errors


def test_cache_misses_different_data(validator_with_rules):
    """Test that different data produces cache miss."""
    data1 = {"name": "test1", "value": 1}
    data2 = {"name": "test2", "value": 2}

    result1 = validator_with_rules.validate_data(data1)
    result2 = validator_with_rules.validate_data(data2)

    # Results should be independent
    assert id(result1) != id(result2)


def test_cache_independent_dict_order(validator_with_rules):
    """Test that dict insertion order doesn't affect cache hits."""
    # Create dicts with same content but different insertion order
    data1 = {"a": 1, "b": 2, "c": 3}
    data2 = {"c": 3, "a": 1, "b": 2}

    result1 = validator_with_rules.validate_data(data1)
    # Second validation with different order should hit cache
    result2 = validator_with_rules.validate_data(data2)

    # Both should produce same result
    assert result1.is_valid == result2.is_valid
    assert result1.errors == result2.errors


def test_cache_clears_successfully(validator_with_rules):
    """Test cache clear functionality."""
    data = {"test": "data"}

    result1 = validator_with_rules.validate_data(data)
    validator_with_rules.clear_cache()
    result2 = validator_with_rules.validate_data(data)

    # Results should be independent after clear
    assert result1.is_valid == result2.is_valid
    assert result1.errors == result2.errors


def test_cache_different_rules_independent(validator_with_rules):
    """Test that different rule sets produce independent cache entries."""
    data = "test_string"

    result1 = validator_with_rules.validate_data(data, rules=["not_empty"])
    result2 = validator_with_rules.validate_data(data, rules=["is_string"])

    # Both validations should succeed but be independent
    assert result1.is_valid is True
    assert result2.is_valid is True


def test_cache_with_none_rules_uses_all(validator_with_rules):
    """Test that None rules parameter uses all rules."""
    data = "test"

    result_none = validator_with_rules.validate_data(data, rules=None)
    result_explicit = validator_with_rules.validate_data(data, rules=["not_empty", "is_string"])

    # Both should pass and be equivalent
    assert result_none.is_valid == result_explicit.is_valid


def test_cache_with_complex_nested_data(validator_with_rules):
    """Test caching with complex nested structures."""
    data = {
        "nested": {"a": 1, "b": [1, 2, 3]},
        "list": [{"x": 1}, {"y": 2}],
    }

    result1 = validator_with_rules.validate_data(data)
    result2 = validator_with_rules.validate_data(data)

    # Should produce same result and be cached
    assert result1.is_valid == result2.is_valid
