"""
Application-wide settings and constants for the Application Factory.
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import os


@dataclass
class AppSettings:
    """
    Application settings and constants.
    """
    
    # Application Info
    APP_NAME: str = "Application Factory"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Craft Perfect Resumes and Cover Letters with AI"
    
    # File Size Limits (in bytes)
    MAX_MASTER_RESUME_SIZE: int = 50 * 1024  # 50KB
    MAX_JOB_DESCRIPTION_SIZE: int = 20 * 1024  # 20KB
    
    # Text Processing Limits
    MIN_MASTER_RESUME_LENGTH: int = 100  # minimum characters
    MIN_JOB_DESCRIPTION_LENGTH: int = 50  # minimum characters
    MAX_MASTER_RESUME_LENGTH: int = 500000  # maximum characters
    MAX_JOB_DESCRIPTION_LENGTH: int = 200000  # maximum characters
    
    # RAG Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    VECTOR_STORE_K: int = 3  # number of chunks to retrieve
    EMBEDDING_MODEL: str = "models/embedding-001"
    
    # LLM Settings
    GEMINI_MODEL: str = "gemini-2.5-flash-preview-05-20"
    MAX_TOKENS: int = 32768
    TEMPERATURE: float = 0.7
    
    # PDF Settings
    PAGE_WIDTH: float = 8.5 * 72  # 8.5 inches in points
    PAGE_HEIGHT: float = 11 * 72  # 11 inches in points
    MARGIN_TOP: float = 36  # 0.5 inch
    MARGIN_BOTTOM: float = 36  # 0.5 inch
    MARGIN_LEFT: float = 36  # 0.5 inch
    MARGIN_RIGHT: float = 36  # 0.5 inch
    
    # Font Settings
    FONT_NAME: str = "Helvetica"
    FONT_SIZE_HEADER: int = 14
    FONT_SIZE_SECTION: int = 12
    FONT_SIZE_BODY: int = 10
    FONT_SIZE_SMALL: int = 9
    
    # File Naming Patterns
    RESUME_FILENAME_PATTERN: str = "{name}_Resume_{company}.pdf"
    COVER_LETTER_FILENAME_PATTERN: str = "{name}_CoverLetter_{company}.pdf"
    FOLDER_NAME_PATTERN: str = "{company}_{position}"
    
    # Supported File Types (using default_factory for mutable defaults)
    SUPPORTED_RESUME_TYPES: list = field(default_factory=lambda: ["txt", "md"])
    
    # Default Paths
    DEFAULT_DATA_DIR: str = "data"
    DEFAULT_VECTOR_STORE_DIR: str = "data/vector_store"
    DEFAULT_OUTPUT_DIR: str = "generated_applications"
    DEFAULT_TEMP_DIR: str = "temp"
    
    # Streamlit UI Settings
    PAGE_TITLE: str = "Application Factory"
    PAGE_ICON: str = "🏭"
    LAYOUT: str = "wide"
    INITIAL_SIDEBAR_STATE: str = "expanded"
    
    # Section Names (for consistent referencing) - using default_factory
    RESUME_SECTIONS: Dict[str, str] = field(default_factory=lambda: {
        "summary": "Professional Summary",
        "skills": "Skills",
        "education": "Education", 
        "experience": "Work Experience",
        "projects": "Projects"
    })
    
    # Cover Letter Sections - using default_factory
    COVER_LETTER_SECTIONS: Dict[str, str] = field(default_factory=lambda: {
        "intro": "Introduction",
        "body": "Body Paragraphs",
        "conclusion": "Conclusion"
    })
    
    # Error Messages - using default_factory
    ERROR_MESSAGES: Dict[str, str] = field(default_factory=lambda: {
        "no_api_key": "Please provide a valid Gemini API key to continue.",
        "no_master_resume": "Please upload your Master Resume to continue.",
        "no_job_description": "Please provide a Job Description to continue.",
        "file_too_large": "File size exceeds the maximum limit.",
        "invalid_file_type": "Unsupported file type. Please use txt or md files.",
        "generation_failed": "Failed to generate content. Please try again.",
        "pdf_too_long": "Generated content exceeds one page. Please shorten the content.",
        "vector_store_error": "Error accessing the vector store. Please re-upload your Master Resume."
    })
    
    # Success Messages - using default_factory
    SUCCESS_MESSAGES: Dict[str, str] = field(default_factory=lambda: {
        "resume_indexed": "✅ Master Resume successfully indexed and ready for use!",
        "resume_generated": "✅ Resume successfully generated!",
        "cover_letter_generated": "✅ Cover Letter successfully generated!",
        "application_complete": "Application documents successfully created and saved!"
    })
    
    @classmethod
    def get_vector_store_path(cls, base_path: str = None) -> str:
        """Get the full path for vector store."""
        if base_path:
            return os.path.join(base_path, "vector_store")
        return cls.DEFAULT_VECTOR_STORE_DIR
    
    @classmethod
    def get_output_path(cls, base_path: str = None) -> str:
        """Get the full path for output directory."""
        if base_path:
            return base_path
        return cls.DEFAULT_OUTPUT_DIR
    
    @classmethod
    def validate_settings(cls) -> bool:
        """Validate that all settings are properly configured."""
        required_settings = [
            cls.CHUNK_SIZE > 0,
            cls.CHUNK_OVERLAP >= 0,
            cls.VECTOR_STORE_K > 0,
            cls.MAX_TOKENS > 0,
            0 <= cls.TEMPERATURE <= 1,
            cls.PAGE_WIDTH > 0,
            cls.PAGE_HEIGHT > 0
        ]
        return all(required_settings) 