"""Centralized logging configuration for AgingResearchAI.

Provides structured logging with optional file output and log rotation.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


# =============================================================================
# Log Formatters
# =============================================================================

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "taskName",
            ):
                log_entry[key] = value

        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# =============================================================================
# Logger Configuration
# =============================================================================

def setup_logging(
    level: str = "INFO",
    log_dir: str | Path | None = None,
    json_format: bool = False,
    console: bool = True,
    file_logging: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """Set up logging configuration for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        json_format: Use JSON format for file logs
        console: Enable console logging
        file_logging: Enable file logging
        max_bytes: Max size per log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Root logger configured for the application
    """
    # Get root logger for our application
    root_logger = logging.getLogger("aging_research_ai")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        # Use colored format for console
        if sys.stdout.isatty():
            console_format = ColoredFormatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        else:
            console_format = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        console_handler.setFormatter(console_format)
        root_logger.addHandler(console_handler)

    # File handler
    if file_logging:
        if log_dir is None:
            log_dir = Path("logs")
        else:
            log_dir = Path(log_dir)

        log_dir.mkdir(parents=True, exist_ok=True)

        # Main log file
        log_file = log_dir / "aging_research_ai.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)

        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))

        root_logger.addHandler(file_handler)

        # Error-only log file
        error_file = log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d\n%(message)s\n",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root_logger.addHandler(error_handler)

    # Also configure third-party loggers
    for logger_name in ["uvicorn", "fastapi", "httpx"]:
        third_party = logging.getLogger(logger_name)
        third_party.setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the application prefix.

    Args:
        name: Logger name (will be prefixed with 'aging_research_ai.')

    Returns:
        Logger instance
    """
    if not name.startswith("aging_research_ai"):
        name = f"aging_research_ai.{name}"
    return logging.getLogger(name)


# =============================================================================
# Context Logging
# =============================================================================

class LogContext:
    """Context manager for adding context to log messages."""

    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self._old_factory = None

    def __enter__(self):
        old_factory = logging.getLogRecordFactory()
        self._old_factory = old_factory
        context = self.context

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            for key, value in context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._old_factory:
            logging.setLogRecordFactory(self._old_factory)
        return False


def log_with_context(logger: logging.Logger, level: int, msg: str, **context):
    """Log a message with additional context fields.

    Args:
        logger: Logger to use
        level: Log level
        msg: Message to log
        **context: Additional context fields
    """
    with LogContext(logger, **context):
        logger.log(level, msg)


# =============================================================================
# Audit Logging
# =============================================================================

class AuditLogger:
    """Specialized logger for audit events."""

    def __init__(self, log_dir: str | Path | None = None):
        self.logger = get_logger("audit")

        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            audit_file = log_dir / "audit.log"
            handler = logging.handlers.RotatingFileHandler(
                audit_file,
                maxBytes=10 * 1024 * 1024,
                backupCount=10,
                encoding="utf-8",
            )
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)

    def log_api_call(
        self,
        endpoint: str,
        method: str,
        user_id: str | None = None,
        status_code: int = 200,
        duration_ms: int = 0,
        **extra,
    ):
        """Log an API call."""
        log_with_context(
            self.logger,
            logging.INFO,
            f"API {method} {endpoint} -> {status_code}",
            event_type="api_call",
            endpoint=endpoint,
            method=method,
            user_id=user_id,
            status_code=status_code,
            duration_ms=duration_ms,
            **extra,
        )

    def log_llm_call(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_ms: int = 0,
        cost_usd: float = 0.0,
        **extra,
    ):
        """Log an LLM API call."""
        log_with_context(
            self.logger,
            logging.INFO,
            f"LLM {model}: {input_tokens} in / {output_tokens} out",
            event_type="llm_call",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            cost_usd=cost_usd,
            **extra,
        )

    def log_debate(
        self,
        topic: str,
        rounds: int,
        consensus_count: int,
        rejected_count: int,
        duration_ms: int = 0,
        **extra,
    ):
        """Log a debate session."""
        log_with_context(
            self.logger,
            logging.INFO,
            f"Debate on '{topic}': {consensus_count} consensus, {rejected_count} rejected",
            event_type="debate",
            topic=topic,
            rounds=rounds,
            consensus_count=consensus_count,
            rejected_count=rejected_count,
            duration_ms=duration_ms,
            **extra,
        )


# =============================================================================
# Default Setup
# =============================================================================

# Configure logging on module import if not already configured
_default_logger = None


def ensure_logging():
    """Ensure logging is configured."""
    global _default_logger
    if _default_logger is None:
        level = os.getenv("LOG_LEVEL", "INFO")
        log_dir = os.getenv("LOG_DIR", "logs")
        json_format = os.getenv("LOG_JSON", "false").lower() == "true"

        _default_logger = setup_logging(
            level=level,
            log_dir=log_dir,
            json_format=json_format,
        )
    return _default_logger
