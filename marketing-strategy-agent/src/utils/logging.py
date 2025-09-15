"""
Centralized logging configuration for the Marketing Strategy Agent.

This module provides a unified logging setup with structured logging support,
multiple output formats, and proper configuration for different environments.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

import structlog
from pythonjsonlogger import jsonlogger

from config import settings


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured data."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "thread": record.thread,
            "thread_name": record.threadName,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        extra_fields = {
            key: value for key, value in record.__dict__.items()
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'exc_info', 'exc_text',
                'stack_info', 'getMessage'
            }
        }
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class ComponentLogger:
    """Logger for specific components with predefined context."""
    
    def __init__(self, component: str, logger: logging.Logger):
        """Initialize component logger."""
        self.component = component
        self.logger = logger
    
    def _log(self, level: int, message: str, **kwargs):
        """Log message with component context."""
        extra = {
            "component": self.component,
            **kwargs
        }
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message."""
        if exc_info:
            import traceback
            kwargs["traceback"] = traceback.format_exc()
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, exc_info: bool = False, **kwargs):
        """Log critical message."""
        if exc_info:
            import traceback
            kwargs["traceback"] = traceback.format_exc()
        self._log(logging.CRITICAL, message, **kwargs)


def setup_logging() -> None:
    """Setup centralized logging configuration."""
    # Create logs directory
    log_dir = Path(settings.log_file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set log level
    log_level = getattr(logging, settings.log_level)
    root_logger.setLevel(log_level)
    
    if settings.structured_logging and settings.log_format == "json":
        # Use structured logging with JSON format
        formatter = StructuredFormatter()
        
        # File handler for JSON logs
        file_handler = logging.FileHandler(settings.log_file_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        
        # Console handler for JSON logs
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        
        # Add handlers to root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
    else:
        # Use standard logging format
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # File handler
        file_handler = logging.FileHandler(settings.log_file_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        
        # Add handlers to root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    # Set specific logger levels
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("pymysql").setLevel(logging.WARNING)
    
    # Log setup completion
    logger = logging.getLogger(__name__)
    logger.info("Logging configuration completed", extra={
        "log_level": settings.log_level,
        "structured_logging": settings.structured_logging,
        "log_format": settings.log_format,
        "log_file": settings.log_file_path
    })


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    # Ensure logging is setup
    if not logging.getLogger().handlers:
        setup_logging()
    
    return logging.getLogger(name)


def get_component_logger(component: str, name: str) -> ComponentLogger:
    """Get a component-specific logger."""
    logger = get_logger(name)
    return ComponentLogger(component, logger)


def log_function_entry(logger: logging.Logger, function_name: str, **kwargs):
    """Log function entry with parameters."""
    logger.debug(f"Entering {function_name}", extra={
        "function": function_name,
        "parameters": kwargs,
        "event": "function_entry"
    })


def log_function_exit(logger: logging.Logger, function_name: str, result: Any = None, duration: Optional[float] = None):
    """Log function exit with result."""
    extra = {
        "function": function_name,
        "event": "function_exit"
    }
    
    if result is not None:
        extra["result_type"] = type(result).__name__
    
    if duration is not None:
        extra["duration_ms"] = duration * 1000
    
    logger.debug(f"Exiting {function_name}", extra=extra)


def log_api_call(logger: logging.Logger, method: str, url: str, status_code: int, duration: float, **kwargs):
    """Log API call details."""
    logger.info(f"API call: {method} {url}", extra={
        "api_method": method,
        "api_url": url,
        "status_code": status_code,
        "duration_ms": duration * 1000,
        "event": "api_call",
        **kwargs
    })


def log_database_operation(logger: logging.Logger, operation: str, table: str, duration: float, **kwargs):
    """Log database operation details."""
    logger.info(f"Database operation: {operation} on {table}", extra={
        "db_operation": operation,
        "db_table": table,
        "duration_ms": duration * 1000,
        "event": "database_operation",
        **kwargs
    })


def log_agent_activity(logger: logging.Logger, agent_name: str, activity: str, **kwargs):
    """Log agent activity."""
    logger.info(f"Agent {agent_name}: {activity}", extra={
        "agent_name": agent_name,
        "activity": activity,
        "event": "agent_activity",
        **kwargs
    })


def log_workflow_step(logger: logging.Logger, workflow: str, step: str, status: str, **kwargs):
    """Log workflow step execution."""
    logger.info(f"Workflow {workflow}: {step} - {status}", extra={
        "workflow": workflow,
        "step": step,
        "status": status,
        "event": "workflow_step",
        **kwargs
    })


# Initialize logging on module import
setup_logging()