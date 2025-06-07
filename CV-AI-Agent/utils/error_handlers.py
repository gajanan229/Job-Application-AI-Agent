"""Error handling utilities for the Application Factory."""

import functools
import traceback
from typing import Any, Callable, Optional, Type, Union
import streamlit as st

from config.logging_config import get_logger
from .session_utils import SessionManager

logger = get_logger(__name__)


class ApplicationError(Exception):
    """Base exception class for Application Factory errors."""
    
    def __init__(self, message: str, details: Optional[str] = None, user_message: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.details = details
        self.user_message = user_message or message


class FileProcessingError(ApplicationError):
    """Error in file processing operations."""
    pass


class LLMError(ApplicationError):
    """Error in LLM operations."""
    pass


class RAGError(ApplicationError):
    """Error in RAG operations."""
    pass


class DocumentGenerationError(ApplicationError):
    """Error in document generation."""
    pass


class ConfigurationError(ApplicationError):
    """Error in configuration."""
    pass


class ValidationError(ApplicationError):
    """Error in validation."""
    pass


class ErrorHandler:
    """Centralized error handling for the Application Factory."""
    
    @staticmethod
    def handle_error(
        error: Exception,
        context: str = "",
        show_user_message: bool = True,
        reraise: bool = False
    ) -> Optional[str]:
        """
        Handle an error with appropriate logging and user notification.
        
        Args:
            error: The exception that occurred
            context: Context information about where the error occurred
            show_user_message: Whether to show error message to user
            reraise: Whether to reraise the exception after handling
        
        Returns:
            User-friendly error message
        """
        try:
            # Get error details
            error_type = type(error).__name__
            error_message = str(error)
            
            # Create context string
            full_context = f"{context}: " if context else ""
            log_message = f"{full_context}{error_type}: {error_message}"
            
            # Log the error with traceback
            logger.error(log_message)
            logger.debug(traceback.format_exc())
            
            # Determine user message
            if isinstance(error, ApplicationError):
                user_message = error.user_message
            else:
                user_message = ErrorHandler._get_user_friendly_message(error, context)
            
            # Show message to user if requested
            if show_user_message and user_message:
                SessionManager.add_error_message(user_message)
                SessionManager.set_processing_status(None)  # Clear processing status
            
            # Reraise if requested
            if reraise:
                raise
            
            return user_message
            
        except Exception as handling_error:
            # If error handling itself fails, log and show generic message
            logger.critical(f"Error in error handler: {handling_error}")
            generic_message = "An unexpected error occurred. Please try again."
            
            if show_user_message:
                SessionManager.add_error_message(generic_message)
                SessionManager.set_processing_status(None)
            
            return generic_message
    
    @staticmethod
    def _get_user_friendly_message(error: Exception, context: str = "") -> str:
        """
        Convert technical error to user-friendly message.
        
        Args:
            error: The exception
            context: Context information
        
        Returns:
            User-friendly error message
        """
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # File-related errors
        if "file" in error_message or "path" in error_message:
            if "not found" in error_message or "no such file" in error_message:
                return "File not found. Please check the file path and try again."
            elif "permission" in error_message:
                return "Permission denied. Please check file permissions."
            elif "size" in error_message:
                return "File size is too large. Please use a smaller file."
            else:
                return "File processing error. Please check your file and try again."
        
        # API-related errors
        if "api" in error_message or "key" in error_message or "authentication" in error_message:
            return "API authentication error. Please check your API key configuration."
        
        if "rate" in error_message or "quota" in error_message or "limit" in error_message:
            return "API rate limit exceeded. Please wait a moment and try again."
        
        if "network" in error_message or "connection" in error_message or "timeout" in error_message:
            return "Network connection error. Please check your internet connection and try again."
        
        # PDF-related errors
        if "pdf" in error_message:
            return "PDF processing error. Please ensure the file is a valid PDF document."
        
        # Document generation errors
        if context and "document" in context.lower():
            return "Document generation error. Please try again or contact support."
        
        # Memory-related errors
        if "memory" in error_message or "ram" in error_message:
            return "Insufficient memory. Please try with smaller files or restart the application."
        
        # Generic error based on type
        if error_type in ["ValueError", "TypeError"]:
            return "Invalid data provided. Please check your inputs and try again."
        
        elif error_type in ["ConnectionError", "TimeoutError"]:
            return "Connection error. Please check your internet connection and try again."
        
        elif error_type == "PermissionError":
            return "Permission denied. Please check file permissions or try running as administrator."
        
        else:
            return f"An error occurred during {context.lower() if context else 'processing'}. Please try again."
    
    @staticmethod
    def safe_execute(
        func: Callable,
        *args,
        context: str = "",
        default_return: Any = None,
        show_error: bool = True,
        **kwargs
    ) -> Any:
        """
        Safely execute a function with error handling.
        
        Args:
            func: Function to execute
            *args: Arguments for the function
            context: Context description for error messages
            default_return: Value to return if function fails
            show_error: Whether to show error to user
            **kwargs: Keyword arguments for the function
        
        Returns:
            Function result or default_return if error occurs
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            ErrorHandler.handle_error(
                e, 
                context=context or f"executing {func.__name__}",
                show_user_message=show_error
            )
            return default_return


def error_handler(
    context: str = "",
    show_user_message: bool = True,
    reraise: bool = False,
    default_return: Any = None
):
    """
    Decorator for automatic error handling.
    
    Args:
        context: Context description for error messages
        show_user_message: Whether to show error to user
        reraise: Whether to reraise exceptions after handling
        default_return: Value to return if function fails (only if reraise=False)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ErrorHandler.handle_error(
                    e,
                    context=context or f"in {func.__name__}",
                    show_user_message=show_user_message,
                    reraise=reraise
                )
                
                if not reraise:
                    return default_return
                
        return wrapper
    return decorator


def streamlit_error_handler(func: Callable) -> Callable:
    """
    Decorator specifically for Streamlit functions to handle errors gracefully.
    Shows errors in the Streamlit UI and doesn't reraise.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Handle the error
            user_message = ErrorHandler.handle_error(
                e,
                context=f"in {func.__name__}",
                show_user_message=False  # We'll show it ourselves
            )
            
            # Show error in Streamlit UI
            st.error(f"⚠️ {user_message}")
            
            # Log additional details for debugging
            logger.error(f"Streamlit function {func.__name__} failed: {e}")
            
            return None
    
    return wrapper


def validate_file_operation(func: Callable) -> Callable:
    """
    Decorator for file operations with specific validation.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            raise FileProcessingError(
                f"File not found: {e}",
                details=str(e),
                user_message="The specified file could not be found. Please check the file path."
            )
        except PermissionError as e:
            raise FileProcessingError(
                f"Permission denied: {e}",
                details=str(e),
                user_message="Permission denied. Please check file permissions or try running as administrator."
            )
        except OSError as e:
            raise FileProcessingError(
                f"File system error: {e}",
                details=str(e),
                user_message="File system error. Please check the file and try again."
            )
    
    return wrapper


def validate_api_operation(func: Callable) -> Callable:
    """
    Decorator for API operations with specific validation.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = str(e).lower()
            
            if "api key" in error_message or "authentication" in error_message:
                raise LLMError(
                    f"API authentication failed: {e}",
                    details=str(e),
                    user_message="API authentication failed. Please check your Google API key."
                )
            elif "rate limit" in error_message or "quota" in error_message:
                raise LLMError(
                    f"API rate limit exceeded: {e}",
                    details=str(e),
                    user_message="API rate limit exceeded. Please wait a moment and try again."
                )
            elif "network" in error_message or "connection" in error_message:
                raise LLMError(
                    f"Network error: {e}",
                    details=str(e),
                    user_message="Network connection error. Please check your internet connection."
                )
            else:
                raise LLMError(
                    f"API error: {e}",
                    details=str(e),
                    user_message="An error occurred while communicating with the AI service. Please try again."
                )
    
    return wrapper


# Convenience function for Streamlit error display
def show_error_messages():
    """Display error messages in Streamlit UI."""
    error_messages = SessionManager.get_error_messages()
    
    for error_info in error_messages[-3:]:  # Show last 3 errors
        st.error(f"⚠️ {error_info['message']}")
    
    # Clear old messages after showing
    if error_messages:
        SessionManager.clear_messages()


def show_success_messages():
    """Display success messages in Streamlit UI."""
    success_messages = SessionManager.get_success_messages()
    
    for success_info in success_messages[-3:]:  # Show last 3 successes
        st.success(f"✅ {success_info['message']}")
    
    # Clear old messages after showing
    if success_messages:
        # Keep only error messages, clear success messages
        st.session_state[SessionManager.SUCCESS_MESSAGES_KEY] = [] 