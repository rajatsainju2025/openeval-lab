# Error Handling System - Consolidated

"""
Unified error handling system for OpenEval Lab.

Consolidates error_context.py, error_handling.py, and core/errors.py
into a single, comprehensive error handling module with context,
recovery suggestions, and proper error chains.

Improvements:
- Single source of truth for error types
- Contextual error information
- Recovery suggestions for common errors
- Type-safe error handling
- Error tracking and analytics
"""

from __future__ import annotations

import traceback
from typing import Dict, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from functools import wraps

from .logging import get_logger

logger = get_logger(__name__)


class ErrorCategory(Enum):
    """Categories of errors in OpenEval."""

    CONFIGURATION = "configuration"
    DATA = "data"
    MODEL = "model"
    EVALUATION = "evaluation"
    CACHE = "cache"
    RUNTIME = "runtime"
    VALIDATION = "validation"
    RESOURCE = "resource"
    NETWORK = "network"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Contextual information about an error."""

    category: ErrorCategory
    message: str
    error_code: str
    suggested_fix: str
    documentation_url: str
    affected_component: Optional[str] = None
    user_action: Optional[str] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "message": self.message,
            "error_code": self.error_code,
            "suggested_fix": self.suggested_fix,
            "documentation_url": self.documentation_url,
            "affected_component": self.affected_component,
            "user_action": self.user_action,
            "debug_info": self.debug_info,
        }


class OpenEvalError(Exception):
    """Base exception for OpenEval Lab."""

    def __init__(
        self,
        message: str,
        error_code: str = "E0000",
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.category = category
        self.context = context or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


class ConfigurationError(OpenEvalError):
    """Configuration-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=kwargs.get("error_code", "E1000"),
            category=ErrorCategory.CONFIGURATION,
            context=kwargs.get("context"),
        )


class DataError(OpenEvalError):
    """Data-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=kwargs.get("error_code", "E2000"),
            category=ErrorCategory.DATA,
            context=kwargs.get("context"),
        )


class ModelError(OpenEvalError):
    """Model-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=kwargs.get("error_code", "E3000"),
            category=ErrorCategory.MODEL,
            context=kwargs.get("context"),
        )


class EvaluationError(OpenEvalError):
    """Evaluation-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=kwargs.get("error_code", "E4000"),
            category=ErrorCategory.EVALUATION,
            context=kwargs.get("context"),
        )


class CacheError(OpenEvalError):
    """Cache-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=kwargs.get("error_code", "E5000"),
            category=ErrorCategory.CACHE,
            context=kwargs.get("context"),
        )


class ValidationError(OpenEvalError):
    """Validation-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=kwargs.get("error_code", "E6000"),
            category=ErrorCategory.VALIDATION,
            context=kwargs.get("context"),
        )


class ResourceError(OpenEvalError):
    """Resource-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=kwargs.get("error_code", "E7000"),
            category=ErrorCategory.RESOURCE,
            context=kwargs.get("context"),
        )


class NetworkError(OpenEvalError):
    """Network-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=kwargs.get("error_code", "E8000"),
            category=ErrorCategory.NETWORK,
            context=kwargs.get("context"),
        )


# Error templates with suggested fixes
ERROR_TEMPLATES: Dict[str, ErrorContext] = {
    "E1000": ErrorContext(
        category=ErrorCategory.CONFIGURATION,
        message="Configuration file not found",
        error_code="E1000",
        suggested_fix="Create config file or specify path with --config flag",
        documentation_url="https://docs.openeval.lab/config",
    ),
    "E2000": ErrorContext(
        category=ErrorCategory.DATA,
        message="Dataset loading failed",
        error_code="E2000",
        suggested_fix="Check dataset path and format. See documentation for supported formats.",
        documentation_url="https://docs.openeval.lab/datasets",
    ),
    "E3000": ErrorContext(
        category=ErrorCategory.MODEL,
        message="Model initialization failed",
        error_code="E3000",
        suggested_fix="Verify model name and required dependencies are installed",
        documentation_url="https://docs.openeval.lab/models",
    ),
    "E4000": ErrorContext(
        category=ErrorCategory.EVALUATION,
        message="Evaluation failed",
        error_code="E4000",
        suggested_fix="Check experiment specification and logs for details",
        documentation_url="https://docs.openeval.lab/evaluation",
    ),
    "E5000": ErrorContext(
        category=ErrorCategory.CACHE,
        message="Cache operation failed",
        error_code="E5000",
        suggested_fix="Clear cache and retry. Check disk space and permissions.",
        documentation_url="https://docs.openeval.lab/caching",
    ),
    "E6000": ErrorContext(
        category=ErrorCategory.VALIDATION,
        message="Validation failed",
        error_code="E6000",
        suggested_fix="Verify input data matches expected schema",
        documentation_url="https://docs.openeval.lab/validation",
    ),
    "E7000": ErrorContext(
        category=ErrorCategory.RESOURCE,
        message="Resource limit exceeded",
        error_code="E7000",
        suggested_fix="Increase resource limits, reduce batch size, or process in smaller chunks",
        documentation_url="https://docs.openeval.lab/resources",
    ),
    "E8000": ErrorContext(
        category=ErrorCategory.NETWORK,
        message="Network error",
        error_code="E8000",
        suggested_fix="Check network connection and retry. Check firewall settings.",
        documentation_url="https://docs.openeval.lab/networking",
    ),
}


class ErrorContextFactory:
    """Factory for creating contextualized errors."""

    _templates = ERROR_TEMPLATES

    @classmethod
    def create(
        cls,
        error_code: str,
        message: Optional[str] = None,
        component: Optional[str] = None,
        debug_info: Optional[Dict[str, Any]] = None,
    ) -> ErrorContext:
        """Create an error context."""
        template = cls._templates.get(error_code)
        if template is None:
            # Create generic context
            return ErrorContext(
                category=ErrorCategory.UNKNOWN,
                message=message or "Unknown error",
                error_code=error_code,
                suggested_fix="Check logs and documentation",
                documentation_url="https://docs.openeval.lab/errors",
                debug_info=debug_info or {},
            )

        # Clone template and customize
        context = ErrorContext(
            category=template.category,
            message=message or template.message,
            error_code=error_code,
            suggested_fix=template.suggested_fix,
            documentation_url=template.documentation_url,
            affected_component=component,
            debug_info=debug_info or {},
        )

        return context

    @classmethod
    def add_template(cls, error_code: str, context: ErrorContext) -> None:
        """Add a new error template."""
        cls._templates[error_code] = context


def safe_operation(
    operation_name: str = "operation",
    error_handler: Optional[Callable] = None,
    reraise: bool = False,
) -> Callable:
    """
    Decorator for safe operation execution with error handling.

    Args:
        operation_name: Name of operation for logging
        error_handler: Optional custom error handler
        reraise: Whether to reraise exceptions
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                logger.debug(f"Starting {operation_name}")
                result = func(*args, **kwargs)
                logger.debug(f"Completed {operation_name}")
                return result

            except OpenEvalError as e:
                logger.error(f"OpenEval error in {operation_name}: {e}")
                if error_handler:
                    error_handler(e)
                if reraise:
                    raise
                return None

            except Exception as e:
                logger.error(f"Error in {operation_name}: {e}", exc_info=True)
                if error_handler:
                    error_handler(e)
                if reraise:
                    raise
                return None

        return wrapper

    return decorator


def format_error_message(error: Exception, include_traceback: bool = False) -> str:
    """
    Format error message with context.

    Args:
        error: Exception to format
        include_traceback: Whether to include full traceback

    Returns:
        Formatted error message
    """
    message = str(error)

    if isinstance(error, OpenEvalError):
        message = f"[{error.error_code}] {error.message}"
        if error.context:
            message += f"\nContext: {error.context}"

    if include_traceback:
        message += "\n" + "".join(traceback.format_tb(error.__traceback__))

    return message


def handle_error_gracefully(error: Exception, component: str = "unknown") -> None:
    """
    Handle an error gracefully with logging and recovery suggestions.

    Args:
        error: Exception to handle
        component: Component where error occurred
    """
    logger.error(f"Error in {component}: {format_error_message(error)}")

    if isinstance(error, OpenEvalError):
        context = ErrorContextFactory.create(
            error.error_code, component=component, debug_info=error.context
        )
        logger.info(f"Suggestion: {context.suggested_fix}")
        logger.info(f"Documentation: {context.documentation_url}")
    else:
        logger.info("Enable debug logging for more information")
