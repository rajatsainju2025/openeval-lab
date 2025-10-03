"""Enhanced logging and error handling for OpenEval Lab."""

import logging
import sys
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

try:
    import structlog

    HAS_STRUCTLOG = True
except ImportError:
    structlog = None
    HAS_STRUCTLOG = False


class OpenEvalLogger:
    """Enterprise-grade logging system with structured logging and multiple handlers."""

    def __init__(
        self,
        name: str = "openeval",
        log_dir: Optional[Path] = None,
        log_level: str = "INFO",
        use_structlog: bool = True,
        enable_audit: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
    ):
        """Initialize enterprise logger with multiple handlers and structured logging."""
        self.name = name
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.use_structlog = use_structlog and HAS_STRUCTLOG
        self.enable_audit = enable_audit
        self.audit_log: List[Dict[str, Any]] = []
        self.audit_lock = threading.Lock()

        # Create standard logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Avoid duplicate handlers
        if self.logger.handlers:
            return

        # Configure structured logging if available
        if self.use_structlog:
            self._configure_structlog()
        else:
            self._configure_standard_logging(max_file_size, backup_count)

    def _configure_structlog(self):
        """Configure structlog for structured logging."""
        import structlog

        # Configure structlog processors
        shared_processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]

        if sys.stderr.isatty():
            # Pretty printing for console
            shared_processors.append(structlog.dev.ConsoleRenderer())
        else:
            # JSON for production
            shared_processors.append(structlog.processors.JSONRenderer())

        structlog.configure(
            processors=shared_processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Add file handler for structured logs
        log_file = self.log_dir / f"openeval_structured_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(file_handler)

    def _configure_standard_logging(self, max_file_size: int, backup_count: int):
        """Configure standard logging with multiple handlers."""
        # Rotating file handler for general logs
        log_file = self.log_dir / f"openeval_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_file_size, backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
            )
        )

        # Timed rotating handler for errors
        error_log_file = self.log_dir / "openeval_errors.log"
        error_handler = TimedRotatingFileHandler(
            error_log_file, when="midnight", interval=1, backupCount=30
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s\n"
                "Exception: %(exc_text)s"
            )
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)

    def audit(self, action: str, user: str = "system", resource: str = "", **kwargs):
        """Log audit event for compliance and security tracking."""
        if not self.enable_audit:
            return

        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "user": user,
            "resource": resource,
            "details": kwargs,
        }

        with self.audit_lock:
            self.audit_log.append(audit_entry)

        # Also log to audit file
        audit_file = self.log_dir / "audit.log"
        with open(audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_entry) + "\n")

        # Log to main logger
        self.logger.info(f"AUDIT: {action} by {user} on {resource}", **kwargs)

    def get_audit_trail(
        self, user: Optional[str] = None, action: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get audit trail filtered by user and/or action."""
        with self.audit_lock:
            trail = self.audit_log.copy()

        if user:
            trail = [entry for entry in trail if entry["user"] == user]
        if action:
            trail = [entry for entry in trail if entry["action"] == action]

        return trail

    def export_audit_log(self, filepath: Path) -> None:
        """Export audit log to file."""
        with self.audit_lock:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.audit_log, f, indent=2)

    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.info(message)

    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.debug(message)

    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.warning(message)

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception and structured data."""
        if exception:
            message = f"{message} | Exception: {str(exception)}"
            self.logger.error(message, exc_info=True)
        else:
            if kwargs:
                message = f"{message} | {json.dumps(kwargs)}"
            self.logger.error(message)

    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical message with optional exception and structured data."""
        if exception:
            message = f"{message} | Exception: {str(exception)}"
            self.logger.critical(message, exc_info=True)
        else:
            if kwargs:
                message = f"{message} | {json.dumps(kwargs)}"
            self.logger.critical(message)


class ErrorHandler:
    """Centralized error handling with context tracking."""

    def __init__(self, logger: Optional[OpenEvalLogger] = None):
        """Initialize error handler with optional logger."""
        self.logger = logger or OpenEvalLogger()
        self.error_count = 0
        self.errors: Dict[str, int] = {}

    def handle_error(
        self, error: Exception, context: str = "", critical: bool = False, **metadata
    ) -> None:
        """Handle error with logging and tracking."""
        self.error_count += 1
        error_type = type(error).__name__
        self.errors[error_type] = self.errors.get(error_type, 0) + 1

        error_info = {
            "error_type": error_type,
            "context": context,
            "total_errors": self.error_count,
            "error_count_by_type": self.errors[error_type],
            **metadata,
        }

        if critical:
            self.logger.critical(
                f"Critical error in {context}: {str(error)}", exception=error, **error_info
            )
        else:
            self.logger.error(f"Error in {context}: {str(error)}", exception=error, **error_info)

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        return {
            "total_errors": self.error_count,
            "errors_by_type": self.errors,
            "most_common_error": (
                max(self.errors.items(), key=lambda x: x[1])[0] if self.errors else None
            ),
        }


# Global instances
_global_logger = None
_global_error_handler = None


def get_logger(name: str = "openeval") -> OpenEvalLogger:
    """Get or create global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = OpenEvalLogger(name)
    return _global_logger


def get_error_handler() -> ErrorHandler:
    """Get or create global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler(get_logger())
    return _global_error_handler


def log_execution_time(func):
    """Decorator to log function execution time."""

    def wrapper(*args, **kwargs):
        logger = get_logger()
        start_time = datetime.now()

        try:
            logger.debug(f"Starting execution of {func.__name__}")
            result = func(*args, **kwargs)

            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Completed {func.__name__}", execution_time=execution_time, status="success"
            )
            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            get_error_handler().handle_error(
                e, context=f"function:{func.__name__}", execution_time=execution_time
            )
            raise

    return wrapper
