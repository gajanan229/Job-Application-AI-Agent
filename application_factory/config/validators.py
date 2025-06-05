"""
Input validation utilities for the Application Factory.

Handles validation of master resumes, job descriptions, and other user inputs.
"""

import re
from typing import Dict, List, Tuple, Optional
import logging
from .settings import AppSettings

logger = logging.getLogger(__name__)


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self, is_valid: bool, message: str = "", warnings: List[str] = None):
        self.is_valid = is_valid
        self.message = message
        self.warnings = warnings or []
    
    def __bool__(self):
        return self.is_valid


def validate_text_length(text: str, min_length: int, max_length: int, field_name: str) -> ValidationResult:
    """
    Validate text length within specified bounds.
    
    Args:
        text (str): Text to validate
        min_length (int): Minimum required length
        max_length (int): Maximum allowed length
        field_name (str): Name of the field for error messages
        
    Returns:
        ValidationResult: Validation result
    """
    if not text or not text.strip():
        return ValidationResult(False, f"{field_name} cannot be empty.")
    
    text_length = len(text.strip())
    
    if text_length < min_length:
        return ValidationResult(
            False, 
            f"{field_name} is too short. Minimum {min_length} characters required, got {text_length}."
        )
    
    if text_length > max_length:
        return ValidationResult(
            False,
            f"{field_name} is too long. Maximum {max_length} characters allowed, got {text_length}."
        )
    
    return ValidationResult(True, "Length validation passed.")


def validate_file_content(content: str, max_size_bytes: int, field_name: str) -> ValidationResult:
    """
    Validate file content size.
    
    Args:
        content (str): File content to validate
        max_size_bytes (int): Maximum size in bytes
        field_name (str): Name of the field for error messages
        
    Returns:
        ValidationResult: Validation result
    """
    if not content:
        return ValidationResult(False, f"{field_name} file appears to be empty.")
    
    content_size = len(content.encode('utf-8'))
    
    if content_size > max_size_bytes:
        size_mb = content_size / (1024 * 1024)
        max_mb = max_size_bytes / (1024 * 1024)
        return ValidationResult(
            False,
            f"{field_name} file is too large ({size_mb:.2f}MB). Maximum size: {max_mb:.2f}MB."
        )
    
    return ValidationResult(True, "File size validation passed.")


def check_resume_completeness(resume_content: str) -> ValidationResult:
    """
    Check if resume contains essential sections.
    
    Args:
        resume_content (str): Resume content to analyze
        
    Returns:
        ValidationResult: Validation result with warnings for missing sections
    """
    content_lower = resume_content.lower()
    warnings = []
    
    # Essential sections to look for
    essential_patterns = {
        "contact_info": ["email", "@", "phone", "linkedin", "github"],
        "experience": ["experience", "work", "employment", "intern", "position"],
        "education": ["education", "degree", "university", "college", "school"],
        "skills": ["skills", "technologies", "languages", "tools"]
    }
    
    for section, patterns in essential_patterns.items():
        if not any(pattern in content_lower for pattern in patterns):
            warnings.append(f"Resume may be missing {section.replace('_', ' ')} section.")
    
    # Check for basic structure
    lines = resume_content.strip().split('\n')
    if len(lines) < 10:
        warnings.append("Resume appears to be very short. Consider adding more detail.")
    
    # Check for bullet points or structured content
    if not any(line.strip().startswith(('•', '-', '*', '▪')) for line in lines):
        warnings.append("Resume may benefit from bullet points for better readability.")
    
    return ValidationResult(True, "Resume completeness check completed.", warnings)


def validate_master_resume(content: str) -> ValidationResult:
    """
    Comprehensive validation of master resume content.
    
    Args:
        content (str): Master resume content
        
    Returns:
        ValidationResult: Complete validation result
    """
    # File size validation
    size_result = validate_file_content(
        content, 
        AppSettings.MAX_MASTER_RESUME_SIZE, 
        "Master Resume"
    )
    if not size_result:
        return size_result
    
    # Length validation
    length_result = validate_text_length(
        content,
        AppSettings.MIN_MASTER_RESUME_LENGTH,
        AppSettings.MAX_MASTER_RESUME_LENGTH,
        "Master Resume"
    )
    if not length_result:
        return length_result
    
    # Content quality checks
    completeness_result = check_resume_completeness(content)
    
    # Combine results
    all_warnings = completeness_result.warnings
    
    return ValidationResult(True, "Master Resume validation passed.", all_warnings)


def validate_job_description(content: str) -> ValidationResult:
    """
    Comprehensive validation of job description content.
    
    Args:
        content (str): Job description content
        
    Returns:
        ValidationResult: Complete validation result
    """
    # File size validation
    size_result = validate_file_content(
        content,
        AppSettings.MAX_JOB_DESCRIPTION_SIZE,
        "Job Description"
    )
    if not size_result:
        return size_result
    
    # Length validation
    length_result = validate_text_length(
        content,
        AppSettings.MIN_JOB_DESCRIPTION_LENGTH,
        AppSettings.MAX_JOB_DESCRIPTION_LENGTH,
        "Job Description"
    )
    if not length_result:
        return length_result
    
    # Content quality checks
    content_lower = content.lower()
    warnings = []
    
    # Check for key job description elements
    jd_elements = {
        "responsibilities": ["responsibilities", "duties", "tasks", "role"],
        "requirements": ["requirements", "qualifications", "skills", "experience"],
        "company_info": ["company", "organization", "about us", "we are"]
    }
    
    for element, patterns in jd_elements.items():
        if not any(pattern in content_lower for pattern in patterns):
            warnings.append(f"Job description may be missing {element.replace('_', ' ')}.")
    
    # Check for specific technologies/skills mentioned
    if not re.search(r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)*\b', content):
        warnings.append("Job description may benefit from more specific technical requirements.")
    
    return ValidationResult(True, "Job Description validation passed.", warnings)


def validate_user_name(name: str) -> ValidationResult:
    """
    Validate user name for file naming.
    
    Args:
        name (str): User name to validate
        
    Returns:
        ValidationResult: Validation result
    """
    if not name or not name.strip():
        return ValidationResult(False, "Name cannot be empty.")
    
    # Check for basic safety
    if len(name.strip()) < 2:
        return ValidationResult(False, "Name must be at least 2 characters long.")
    
    if len(name.strip()) > 50:
        return ValidationResult(False, "Name must be 50 characters or less.")
    
    # Check for problematic characters
    if re.search(r'[<>:"/\\|?*]', name):
        return ValidationResult(False, "Name contains invalid characters for file naming.")
    
    return ValidationResult(True, "Name validation passed.")


def validate_output_path(path: str) -> ValidationResult:
    """
    Validate output path for generating files.
    
    Args:
        path (str): Output path to validate
        
    Returns:
        ValidationResult: Validation result
    """
    if not path or not path.strip():
        return ValidationResult(False, "Output path cannot be empty.")
    
    # Basic path safety checks
    normalized_path = path.strip().replace('\\', '/')
    
    # Check for dangerous paths
    dangerous_patterns = [
        '../', '..\\', '/etc/', '/sys/', 'C:\\Windows', 'C:\\Program Files'
    ]
    
    for pattern in dangerous_patterns:
        if pattern.lower() in normalized_path.lower():
            return ValidationResult(False, "Output path appears to be unsafe.")
    
    return ValidationResult(True, "Output path validation passed.")


def validate_streamlit_upload(uploaded_file) -> ValidationResult:
    """
    Validate a Streamlit uploaded file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        ValidationResult: Validation result
    """
    if uploaded_file is None:
        return ValidationResult(False, "No file uploaded.")
    
    # Check file type
    if uploaded_file.type not in ['text/plain', 'text/markdown']:
        return ValidationResult(False, f"Unsupported file type: {uploaded_file.type}. Please upload a .txt or .md file.")
    
    # Check file size
    if uploaded_file.size > AppSettings.MAX_MASTER_RESUME_SIZE:
        size_mb = uploaded_file.size / (1024 * 1024)
        max_mb = AppSettings.MAX_MASTER_RESUME_SIZE / (1024 * 1024)
        return ValidationResult(False, f"File too large ({size_mb:.2f}MB). Maximum: {max_mb:.2f}MB.")
    
    # Check file name
    if not uploaded_file.name or len(uploaded_file.name) > 100:
        return ValidationResult(False, "Invalid file name.")
    
    return ValidationResult(True, "File upload validation passed.")


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """
    Sanitize user input text.
    
    Args:
        text (str): Text to sanitize
        max_length (int): Maximum length to truncate to
        
    Returns:
        str: Sanitized text
    """
    if not text:
        return ""
    
    # Remove potential HTML/script content
    text = re.sub(r'<[^>]*>', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Trim to max length
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text.strip() 