"""Configuration settings for the Application Factory."""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv


@dataclass
class AppConfig:
    """Application configuration class with all settings."""
    
    # API Configuration
    google_api_key: Optional[str] = None
    gemini_model: str = "gemini-pro"
    embedding_model: str = "models/embedding-001"
    
    # Application Configuration
    app_env: str = "development"
    debug: bool = True
    
    # File Configuration
    max_file_size_mb: int = 10
    output_dir: str = "generated_documents"
    temp_dir: str = "generated_documents/temp"
    
    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 5
    
    # LLM Configuration
    llm_model_name: str = "gemini-2.0-flash"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 8192
    llm_top_p: float = 0.95
    llm_top_k: int = 40
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "application_factory.log"
    
    # Optional User Information
    default_company_name: str = ""
    default_user_name: str = ""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._ensure_directories()
    
    def _validate_config(self):
        """Validate configuration values."""
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive")
        
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        if not (0 <= self.llm_temperature <= 2):
            raise ValueError("llm_temperature must be between 0 and 2")
        
        if self.llm_max_tokens <= 0:
            raise ValueError("llm_max_tokens must be positive")
        
        if not (0 <= self.llm_top_p <= 1):
            raise ValueError("llm_top_p must be between 0 and 1")
        
        if self.llm_top_k <= 0:
            raise ValueError("llm_top_k must be positive")
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def load_from_env(cls) -> 'AppConfig':
        """Load configuration from environment variables."""
        load_dotenv()
        
        return cls(
            # API Configuration
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-pro"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "models/embedding-001"),
            
            # Application Configuration
            app_env=os.getenv("APP_ENV", "development"),
            debug=os.getenv("DEBUG", "true").lower() == "true",
            
            # File Configuration
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "10")),
            output_dir=os.getenv("OUTPUT_DIR", "generated_documents"),
            temp_dir=os.getenv("TEMP_DIR", "generated_documents/temp"),
            
            # RAG Configuration
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            retrieval_k=int(os.getenv("RETRIEVAL_K", "5")),
            
            # LLM Configuration
            llm_model_name=os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "8192")),
            llm_top_p=float(os.getenv("LLM_TOP_P", "0.95")),
            llm_top_k=int(os.getenv("LLM_TOP_K", "40")),
            
            # Logging Configuration
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "application_factory.log"),
            
            # Optional User Information
            default_company_name=os.getenv("DEFAULT_COMPANY_NAME", ""),
            default_user_name=os.getenv("DEFAULT_USER_NAME", ""),
        )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            field: getattr(self, field) 
            for field in self.__dataclass_fields__
        }
    
    def is_api_key_configured(self) -> bool:
        """Check if Google API key is configured."""
        return self.google_api_key is not None and self.google_api_key.strip() != ""


# Global configuration instance
config = AppConfig.load_from_env() 