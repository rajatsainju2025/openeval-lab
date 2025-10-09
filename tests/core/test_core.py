"""Unit tests for core module functionality."""

import random

from openeval.core import _categorize_error, _summarize_errors
from openeval.utils import set_seed, hash_file, hash_prompt


def test_error_categorization():
    """Test error categorization function."""
    # Test common error types
    timeout_err = TimeoutError("Operation timed out")
    assert _categorize_error(timeout_err) == "TIMEOUT"

    rate_limit_err = Exception("Rate limit exceeded - 429")
    assert _categorize_error(rate_limit_err) == "RATE_LIMIT"

    network_err = ConnectionError("Network error occurred")
    assert _categorize_error(network_err) == "NETWORK"

    auth_err = Exception("Authentication failed - 401")
    assert _categorize_error(auth_err) == "AUTH"

    server_err = Exception("Internal server error - 500")
    assert _categorize_error(server_err) == "SERVER_ERROR"

    invalid_req_err = ValueError("Invalid request - 400")
    assert _categorize_error(invalid_req_err) == "INVALID_REQUEST"

    # Test unknown error with type name
    class CustomError(Exception):
        pass

    custom_err = CustomError("Custom error")
    assert _categorize_error(custom_err) == "CustomError"


def test_error_summarization():
    """Test error summarization function."""
    errors = [
        "[timeout]Request timed out",
        "[value]Invalid value",
        "[timeout]Another timeout",
        None,
        "[key]Missing key",
        "[timeout]Third timeout",
    ]

    summary = _summarize_errors(errors)
    assert isinstance(summary, dict)
    assert summary["timeout"] == 3
    assert summary["value"] == 1
    assert summary["key"] == 1


def test_example_dataclass():
    """Test Example dataclass functionality."""
    from openeval.core import Example

    # Test basic initialization with only required fields
    example = Example(
        id="test1",
        input="What is 2+2?",
        reference="4",
    )
    assert example.id == "test1"
    assert example.input == "What is 2+2?"
    assert example.reference == "4"

    # Test with optional meta field
    example_with_meta = Example(
        id="test2", input="What is 2+2?", reference="4", meta={"difficulty": "easy"}
    )
    assert example_with_meta.meta and example_with_meta.meta["difficulty"] == "easy"


# Remove these functions as they've been replaced by the updated versions below


def test_hash_prompt():
    """Test prompt hashing function."""
    # Test basic hashing
    prompt_parts = ["Test prompt", "model1", "adapter1"]
    hash1 = hash_prompt(prompt_parts)
    assert isinstance(hash1, str)
    assert len(hash1) > 0

    # Test consistency
    hash2 = hash_prompt(prompt_parts)
    assert hash1 == hash2

    # Test different prompts have different hashes
    different_parts = ["Different prompt", "model1", "adapter1"]
    hash3 = hash_prompt(different_parts)
    assert hash1 != hash3

    # Test empty list
    empty_parts = []
    hash_empty = hash_prompt(empty_parts)
    assert isinstance(hash_empty, str)


def test_set_seed():
    """Test seed setting function."""
    # Test with specific seed
    set_seed(42)

    # Test with None seed (should not raise errors)
    set_seed(None)

    # Test seeding affects randomness
    set_seed(42)
    rand1 = random.random()
    set_seed(42)
    rand2 = random.random()
    assert rand1 == rand2


def test_hash_file(tmp_path):
    """Test file hashing function."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_content = "Hello, world!"
    test_file.write_text(test_content)

    # Test basic hashing
    hash1 = hash_file(test_file)
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA-256 produces 64 char hex string

    # Test consistency
    hash2 = hash_file(test_file)
    assert hash1 == hash2

    # Test with different content
    test_file.write_text("Different content")
    hash3 = hash_file(test_file)
    assert hash1 != hash3

    # Test with string path
    hash4 = hash_file(str(test_file))
    assert hash3 == hash4  # Should work same with Path or str

    # Test with custom algorithm
    hash_md5 = hash_file(test_file, algo="md5")
    assert len(hash_md5) == 32  # MD5 produces 32 char hex string
