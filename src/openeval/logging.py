"""Enhanced logging and error handling for OpenEval Lab."""

import logging
import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class OpenEvalLogger:
    """Centralized logging system for OpenEval Lab."""
    
    def __init__(self, name: str = "openeval", log_dir: Optional[Path] = None):
        """Initialize logger with file and console handlers."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if self.logger.handlers:
            return
            
        # Create log directory
        if log_dir is None:
            log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # File handler
        log_file = log_dir / f"openeval_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
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
        self, 
        error: Exception, 
        context: str = "",
        critical: bool = False,
        **metadata
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
            **metadata
        }
        
        if critical:
            self.logger.critical(
                f"Critical error in {context}: {str(error)}", 
                exception=error, 
                **error_info
            )
        else:
            self.logger.error(
                f"Error in {context}: {str(error)}", 
                exception=error, 
                **error_info
            )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        return {
            "total_errors": self.error_count,
            "errors_by_type": self.errors,
            "most_common_error": max(self.errors.items(), key=lambda x: x[1])[0] if self.errors else None
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
                f"Completed {func.__name__}", 
                execution_time=execution_time,
                status="success"
            )
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            get_error_handler().handle_error(
                e, 
                context=f"function:{func.__name__}",
                execution_time=execution_time
            )
            raise
    
    return wrapper
