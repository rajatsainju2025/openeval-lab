"""Error handling utilities for the evaluation framework."""

from __future__ import annotations

from typing import Dict, List, Optional


def _categorize_error(err: Exception) -> str:
    """Categorize an exception into standardized error types for consistent reporting.

    Args:
        err: The exception to categorize.

    Returns:
        A standardized error category string, one of:
        - TIMEOUT: Operation timed out
        - RATE_LIMIT: Rate limit exceeded (HTTP 429)
        - NETWORK: Network/connection issues
        - AUTH: Authentication failures (HTTP 401/403)
        - QUOTA: Resource quota exceeded (HTTP 402)
        - SERVER_ERROR: Server-side errors (HTTP 500/502/503)
        - INVALID_REQUEST: Invalid request (HTTP 400)
        - {Exception.__name__}: Other exceptions, using exception type name
    """
    err_str = str(err).lower()
    err_type = type(err).__name__

    if "timeout" in err_str or "timed out" in err_str or isinstance(err, TimeoutError):
        return "TIMEOUT"
    elif "rate limit" in err_str or "429" in err_str:
        return "RATE_LIMIT"
    elif "connection" in err_str or "network" in err_str:
        return "NETWORK"
    elif "authentication" in err_str or "401" in err_str or "403" in err_str:
        return "AUTH"
    elif "quota" in err_str or "402" in err_str:
        return "QUOTA"
    elif "server" in err_str or "500" in err_str or "502" in err_str or "503" in err_str:
        return "SERVER_ERROR"
    elif "invalid" in err_str or "400" in err_str:
        return "INVALID_REQUEST"
    else:
        return f"{err_type}"


def _summarize_errors(per_error: List[Optional[str]]) -> Dict[str, int]:
    """Count and summarize errors by their category.

    Takes a list of error messages (potentially including Nones) and produces a count
    by error category. Error categories are expected to be in the format [CATEGORY]message.

    Args:
        per_error: List of error messages, where each message may be None or a string.
                  Strings starting with [CATEGORY] will be counted under that category.

    Returns:
        A dictionary mapping error categories to their counts. Unknown categories are
        counted under the "UNKNOWN" key.

    Example:
        >>> _summarize_errors(["[TIMEOUT]Request timed out", None, "[TIMEOUT]Another timeout"])
        {'TIMEOUT': 2}
    """
    error_counts: Dict[str, int] = {}
    for error in per_error:
        if error:
            # Extract category from [CATEGORY] message format
            if error.startswith("[") and "]" in error:
                category = error.split("]")[0][1:]
            else:
                category = "UNKNOWN"
            error_counts[category] = error_counts.get(category, 0) + 1
    return error_counts


__all__ = [
    "_categorize_error",
    "_summarize_errors",
]
