import logging
import sys
from datetime import datetime
from typing import Optional

class LoggerConfig:
    def __init__(self, name: str = "rag_chatbot", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Prevent adding multiple handlers if logger already exists
        if not self.logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)

            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)

            # Add handler to logger
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger

# Create a global logger instance
logger_config = LoggerConfig()
logger = logger_config.get_logger()

def sanitize_log_data(data: dict) -> dict:
    """
    Sanitize sensitive data from log entries
    """
    sanitized = data.copy()

    # Remove or mask sensitive fields
    sensitive_fields = ['api_key', 'token', 'password', 'secret', 'key', 'credential',
                       'question', 'context', 'answer', 'content', 'text']

    for key, value in sanitized.items():
        if isinstance(key, str) and any(field in key.lower() for field in sensitive_fields):
            # For questions, answers, and content, we can log length instead of the content
            if 'question' in key.lower() or 'context' in key.lower() or 'answer' in key.lower() or 'content' in key.lower():
                if isinstance(value, str):
                    sanitized[key] = f"[CONTENT OF LENGTH {len(value)} CHARACTERS]"
            else:
                sanitized[key] = "***REDACTED***"
        elif isinstance(value, str) and any(field in key.lower() for field in sensitive_fields):
            sanitized[key] = "***REDACTED***"

    return sanitized

def log_info(message: str, extra: Optional[dict] = None):
    """Log an info message"""
    if extra:
        sanitized_extra = sanitize_log_data(extra)
        logger.info(message, extra=sanitized_extra)
    else:
        logger.info(message)

def log_warning(message: str, extra: Optional[dict] = None):
    """Log a warning message"""
    if extra:
        sanitized_extra = sanitize_log_data(extra)
        logger.warning(message, extra=sanitized_extra)
    else:
        logger.warning(message)

def log_error(message: str, extra: Optional[dict] = None):
    """Log an error message"""
    if extra:
        sanitized_extra = sanitize_log_data(extra)
        logger.error(message, extra=sanitized_extra)
    else:
        logger.error(message)

def log_debug(message: str, extra: Optional[dict] = None):
    """Log a debug message"""
    if extra:
        sanitized_extra = sanitize_log_data(extra)
        logger.debug(message, extra=sanitized_extra)
    else:
        logger.debug(message)

# Convenience function for timing operations
def log_execution_time(func_name: str, start_time: datetime, end_time: datetime):
    """Log execution time of a function"""
    duration = (end_time - start_time).total_seconds()
    log_info(f"{func_name} executed in {duration:.2f} seconds", extra={"duration": duration})