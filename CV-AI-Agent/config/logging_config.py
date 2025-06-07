"""Logging configuration for the Application Factory."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from .settings import config


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for the Application Factory.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, uses config.log_file)
        console_output: Whether to output logs to console
    
    Returns:
        Configured logger instance
    """
    # Use config values if not provided
    log_level = log_level or config.log_level
    log_file = log_file or config.log_file
    
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger("application_factory")
    logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler (if log_file is specified)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        # Use simple formatter for console if not in debug mode
        if numeric_level <= logging.DEBUG:
            console_handler.setFormatter(detailed_formatter)
        else:
            console_handler.setFormatter(simple_formatter)
        
        logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = "application_factory") -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__ from calling module)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def setup_streamlit_logging() -> logging.Logger:
    """
    Set up logging specifically for Streamlit applications.
    Reduces verbosity of Streamlit's internal logging.
    
    Returns:
        Configured logger instance
    """
    # Reduce Streamlit's logging verbosity
    logging.getLogger("streamlit").setLevel(logging.WARNING)
    logging.getLogger("streamlit.runtime").setLevel(logging.WARNING)
    logging.getLogger("streamlit.web").setLevel(logging.WARNING)
    
    # Set up our application logging
    return setup_logging(console_output=True)


def log_function_call(func):
    """
    Decorator to log function calls with parameters and return values.
    Useful for debugging.
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function entry
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned: {type(result)}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
            raise
    
    return wrapper


def log_processing_step(step_name: str):
    """
    Decorator to log processing steps with timing information.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            logger = get_logger(func.__module__)
            
            logger.info(f"Starting: {step_name}")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                logger.info(f"Completed: {step_name} (took {duration:.2f}s)")
                return result
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                logger.error(f"Failed: {step_name} after {duration:.2f}s - {type(e).__name__}: {e}")
                raise
        
        return wrapper
    return decorator


def timing_decorator(func):
    """
    Decorator to log function execution timing.
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        start_time = time.time()
        logger.debug(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logger.debug(f"{func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {type(e).__name__}: {e}")
            raise
    
    return wrapper


# Initialize default logger when module is imported
default_logger = setup_logging() 