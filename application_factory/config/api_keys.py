"""
Secure API key management for the Application Factory.

This module provides secure retrieval of API keys from multiple sources
with proper fallback handling and no storage in application state.
"""

import streamlit as st
import os
from typing import Optional
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_gemini_api_key() -> str:
    """
    Securely retrieve Gemini API key from multiple sources.
    
    Priority order:
    1. Streamlit secrets (.streamlit/secrets.toml)
    2. Environment variables
    3. User input (session state only)
    
    Returns:
        str: The API key if found, empty string otherwise
        
    Raises:
        ValueError: If no API key is found from any source
    """
    # Priority 1: Streamlit secrets
    try:
        if hasattr(st, 'secrets') and "gemini_api_key" in st.secrets:
            logger.info("Using Gemini API key from Streamlit secrets")
            return st.secrets["gemini_api_key"]
    except Exception as e:
        logger.debug(f"Could not access Streamlit secrets: {e}")
    
    # Priority 2: Environment variables
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        logger.info("Using Gemini API key from environment variable")
        return env_key
    
    # Priority 3: User input (session state)
    if hasattr(st, 'session_state'):
        session_key = st.session_state.get("gemini_api_key_input", "")
        if session_key:
            logger.info("Using Gemini API key from user input")
            return session_key
    
    # No key found
    logger.warning("No Gemini API key found from any source")
    return ""


def validate_api_key(api_key: str) -> bool:
    """
    Validate the format of a Gemini API key.
    
    Args:
        api_key (str): The API key to validate
        
    Returns:
        bool: True if the API key format appears valid
    """
    if not api_key:
        return False
    
    # Basic format validation for Google API keys
    # They typically start with 'AIza' and are 39 characters long
    if api_key.startswith('AIza') and len(api_key) == 39:
        return True
    
    # Some API keys might have different formats, so be flexible
    if len(api_key) >= 20:  # Minimum reasonable length
        return True
    
    return False


def get_validated_api_key() -> str:
    """
    Get and validate the Gemini API key.
    
    Returns:
        str: Valid API key
        
    Raises:
        ValueError: If no valid API key is found
    """
    api_key = get_gemini_api_key()
    
    if not validate_api_key(api_key):
        raise ValueError(
            "Invalid or missing Gemini API key. Please provide a valid API key "
            "through Streamlit secrets, environment variables, or user input."
        )
    
    return api_key


def mask_api_key(api_key: str, show_chars: int = 4) -> str:
    """
    Mask an API key for safe display in logs or UI.
    
    Args:
        api_key (str): The API key to mask
        show_chars (int): Number of characters to show at the start
        
    Returns:
        str: Masked API key (e.g., "AIza****")
    """
    if not api_key:
        return "[No API Key]"
    
    if len(api_key) <= show_chars:
        return "*" * len(api_key)
    
    return api_key[:show_chars] + "*" * (len(api_key) - show_chars) 