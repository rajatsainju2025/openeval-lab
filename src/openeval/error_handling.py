"""
Error Handling Utilities for OpenEval Lab.

This module provides centralized error handling, categorization, and recovery
suggestion utilities for consistent error management across the framework.

Key Features:
- Error categorization for standardized reporting
- Recovery suggestions for common error types
- Context-rich error messages for better debugging
- Retry-friendly error classification

Design Goals:
- Users should understand what went wrong and how to fix it
- Errors should be actionable, not just informative
- Consistent error formatting across the entire framework
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ErrorCategory(Enum):
    """Standardized error categories for consistent reporting."""

    TIMEOUT = "TIMEOUT"
    RATE_LIMIT = "RATE_LIMIT"
    NETWORK = "NETWORK"
    AUTH = "AUTH"
    QUOTA = "QUOTA"
    SERVER_ERROR = "SERVER_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    VALIDATION = "VALIDATION"
    CONFIGURATION = "CONFIGURATION"
    RESOURCE = "RESOURCE"
    UNKNOWN = "UNKNOWN"


# Recovery suggestions for each error category
RECOVERY_SUGGESTIONS: Dict[ErrorCategory, List[str]] = {
    ErrorCategory.TIMEOUT: [
        "Increase the timeout value in your configuration",
        "Check if the remote service is responding slowly",
        "Consider reducing batch size for faster processing",
        "Try running during off-peak hours",
    ],
    ErrorCategory.RATE_LIMIT: [
        "Reduce concurrency (--concurrency flag)",
        "Add delays between requests (retry_delay_seconds in config)",
        "Check your API rate limits with the provider",
        "Consider using caching to reduce API calls",
    ],
    ErrorCategory.NETWORK: [
        "Check your internet connection",
        "Verify the API endpoint URL is correct",
        "Check if a firewall is blocking the connection",
        "Try using a VPN if the service is geo-restricted",
    ],
    ErrorCategory.AUTH: [
        "Verify your API key is correct and not expired",
        "Check environment variable OPENAI_API_KEY (or provider-specific key)",
        "Ensure your account has the required permissions",
        "Regenerate your API key if issues persist",
    ],
    ErrorCategory.QUOTA: [
        "Check your API usage quota with the provider",
        "Consider upgrading your API plan",
        "Use caching to reduce redundant API calls",
        "Implement a smaller evaluation batch",
    ],
    ErrorCategory.SERVER_ERROR: [
        "The remote server is experiencing issues - try again later",
        "Check the provider's status page for outages",
        "Implement retry logic with exponential backoff",
        "Consider using a fallback adapter",
    ],
    ErrorCategory.INVALID_REQUEST: [
        "Check your input data format and encoding",
        "Verify the prompt doesn't exceed the model's context limit",
        "Ensure all required fields are provided in the spec",
        "Validate your JSON/YAML configuration syntax",
    ],
    ErrorCategory.VALIDATION: [
        "Review the spec file against the schema (openeval validate <spec>)",
        "Check for typos in field names",
        "Ensure all referenced files exist",
        "Run 'openeval doctor' to diagnose issues",
    ],
    ErrorCategory.CONFIGURATION: [
        "Review your configuration file syntax",
        "Check that all required fields are present",
        "Verify file paths are correct and accessible",
        "Try 'openeval validate <spec>' for detailed validation",
    ],
    ErrorCategory.RESOURCE: [
        "Close other memory-intensive applications",
        "Reduce batch size or concurrency",
        "Enable streaming mode for large datasets",
        "Consider running on a machine with more resources",
    ],
    ErrorCategory.UNKNOWN: [
        "Check the error message for specific details",
        "Search the documentation for similar issues",
        "Enable verbose logging (--verbose flag)",
        "Report the issue on GitHub with full error output",
    ],
}


@dataclass
class ErrorContext:
    """Rich error context for debugging and recovery."""

    category: ErrorCategory
    message: str
    original_exception: Optional[Exception] = None
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Populate suggestions if not provided."""
        if not self.suggestions:
            self.suggestions = RECOVERY_SUGGESTIONS.get(self.category, [])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "category": self.category.value,
            "message": self.message,
            "suggestions": self.suggestions,
            "metadata": self.metadata,
        }

    def format_message(self, include_suggestions: bool = True) -> str:
        """Format error message with optional recovery suggestions."""
        lines = [f"[{self.category.value}] {self.message}"]

        if include_suggestions and self.suggestions:
            lines.append("\nRecovery suggestions:")
            for i, suggestion in enumerate(self.suggestions[:3], 1):
                lines.append(f"  {i}. {suggestion}")

        return "\n".join(lines)


def categorize_error(err: Exception) -> ErrorCategory:
    """Categorize an exception into standardized error types.

    Args:
        err: The exception to categorize.

    Returns:
        An ErrorCategory enum value for consistent error handling.
    """
    err_str = str(err).lower()
    err_type = type(err).__name__.lower()

    # Check for specific error patterns
    if "timeout" in err_str or "timed out" in err_str or isinstance(err, TimeoutError):
        return ErrorCategory.TIMEOUT

    if "rate limit" in err_str or "429" in err_str or "ratelimit" in err_type:
        return ErrorCategory.RATE_LIMIT

    if any(x in err_str for x in ["connection", "network", "dns", "socket"]):
        return ErrorCategory.NETWORK

    if any(x in err_str for x in ["authentication", "401", "403", "unauthorized"]):
        return ErrorCategory.AUTH

    if "quota" in err_str or "402" in err_str:
        return ErrorCategory.QUOTA

    if any(x in err_str for x in ["server", "500", "502", "503", "504"]):
        return ErrorCategory.SERVER_ERROR

    if "invalid" in err_str or "400" in err_str or "bad request" in err_str:
        return ErrorCategory.INVALID_REQUEST

    if "validation" in err_str or "schema" in err_str:
        return ErrorCategory.VALIDATION

    if "config" in err_str or "configuration" in err_str:
        return ErrorCategory.CONFIGURATION

    if any(x in err_str for x in ["memory", "resource", "oom", "out of memory"]):
        return ErrorCategory.RESOURCE

    return ErrorCategory.UNKNOWN


def create_error_context(
    err: Exception,
    additional_context: Optional[Dict[str, Any]] = None,
) -> ErrorContext:
    """Create rich error context from an exception.

    Args:
        err: The exception to contextualize.
        additional_context: Optional additional metadata.

    Returns:
        An ErrorContext with category, message, and recovery suggestions.
    """
    category = categorize_error(err)
    metadata = additional_context or {}
    metadata["exception_type"] = type(err).__name__

    return ErrorContext(
        category=category,
        message=str(err),
        original_exception=err,
        metadata=metadata,
    )


def is_retryable(category: ErrorCategory) -> bool:
    """Check if an error category is typically retryable.

    Args:
        category: The error category to check.

    Returns:
        True if the error is typically retryable with backoff.
    """
    retryable = {
        ErrorCategory.TIMEOUT,
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.NETWORK,
        ErrorCategory.SERVER_ERROR,
    }
    return category in retryable


def should_abort(category: ErrorCategory) -> bool:
    """Check if an error category should abort the entire operation.

    Args:
        category: The error category to check.

    Returns:
        True if the error should abort (not recoverable).
    """
    abort_categories = {
        ErrorCategory.AUTH,
        ErrorCategory.QUOTA,
        ErrorCategory.CONFIGURATION,
    }
    return category in abort_categories
