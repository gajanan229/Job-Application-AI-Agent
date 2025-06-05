"""
Path management utilities for the Application Factory.

Handles creation, validation, and management of file paths and directories.
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PathManager:
    """
    Manages file paths and directory operations for the Application Factory.
    """
    
    def __init__(self, base_output_path: str = "generated_applications"):
        """
        Initialize the PathManager with a base output path.
        
        Args:
            base_output_path (str): Base directory for generated applications
        """
        self.base_output_path = Path(base_output_path)
        self.data_path = Path("data")
        self.vector_store_path = self.data_path / "vector_store"
        self.temp_path = Path("temp")
        
        # Ensure base directories exist
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.base_output_path,
            self.data_path,
            self.vector_store_path,
            self.temp_path
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created/verified directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                raise
    
    def sanitize_filename(self, name: str) -> str:
        """
        Sanitize a string to be safe for use as a filename.
        
        Args:
            name (str): The name to sanitize
            
        Returns:
            str: Sanitized filename
        """
        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
        
        # Remove multiple consecutive underscores and dots
        sanitized = re.sub(r'[_.]{2,}', '_', sanitized)
        
        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip(' .')
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = "unnamed"
        
        # Limit length to avoid filesystem issues
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        
        return sanitized
    
    def extract_company_and_position(self, job_description: str) -> Tuple[str, str]:
        """
        Extract company name and position from job description.
        
        Args:
            job_description (str): The job description text
            
        Returns:
            Tuple[str, str]: (company_name, position_title)
        """
        lines = job_description.strip().split('\n')
        
        # Default values
        company_name = "Company"
        position_title = "Position"
        
        # Try to extract company and position from first few lines
        for i, line in enumerate(lines[:10]):  # Check first 10 lines
            line = line.strip()
            
            # Look for common patterns
            if any(keyword in line.lower() for keyword in ['company:', 'organization:', 'employer:']):
                company_name = line.split(':', 1)[-1].strip()
                break
            elif any(keyword in line.lower() for keyword in ['position:', 'title:', 'role:', 'job:']):
                position_title = line.split(':', 1)[-1].strip()
            elif i == 0 and len(line) > 10:  # First line might be the position
                position_title = line
            elif i == 1 and len(line) > 5:  # Second line might be the company
                company_name = line
        
        # Clean and sanitize
        company_name = self.sanitize_filename(company_name)
        position_title = self.sanitize_filename(position_title)
        
        return company_name, position_title
    
    def create_job_folder(self, job_description: str, custom_name: Optional[str] = None) -> Path:
        """
        Create a job-specific folder for storing application documents.
        
        Args:
            job_description (str): The job description to extract info from
            custom_name (Optional[str]): Custom folder name override
            
        Returns:
            Path: Path to the created job folder
        """
        if custom_name:
            folder_name = self.sanitize_filename(custom_name)
        else:
            company, position = self.extract_company_and_position(job_description)
            folder_name = f"{company}_{position}"
        
        # Ensure unique folder name
        job_folder = self.base_output_path / folder_name
        counter = 1
        original_name = folder_name
        
        while job_folder.exists():
            folder_name = f"{original_name}_{counter}"
            job_folder = self.base_output_path / folder_name
            counter += 1
        
        try:
            job_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created job folder: {job_folder}")
            return job_folder
        except Exception as e:
            logger.error(f"Failed to create job folder {job_folder}: {e}")
            raise
    
    def get_resume_path(self, job_folder: Path, user_name: str = "Resume") -> Path:
        """
        Get the full path for a resume PDF.
        
        Args:
            job_folder (Path): The job-specific folder
            user_name (str): User's name for the filename
            
        Returns:
            Path: Full path for the resume PDF
        """
        sanitized_name = self.sanitize_filename(user_name)
        filename = f"{sanitized_name}_Resume.pdf"
        return job_folder / filename
    
    def get_cover_letter_path(self, job_folder: Path, user_name: str = "CoverLetter") -> Path:
        """
        Get the full path for a cover letter PDF.
        
        Args:
            job_folder (Path): The job-specific folder
            user_name (str): User's name for the filename
            
        Returns:
            Path: Full path for the cover letter PDF
        """
        sanitized_name = self.sanitize_filename(user_name)
        filename = f"{sanitized_name}_CoverLetter.pdf"
        return job_folder / filename
    
    def get_temp_pdf_path(self, prefix: str = "temp") -> Path:
        """
        Get a temporary PDF path for previews.
        
        Args:
            prefix (str): Prefix for the temp file
            
        Returns:
            Path: Path to temporary PDF file
        """
        filename = f"{prefix}_{os.getpid()}.pdf"
        return self.temp_path / filename
    
    def get_vector_store_path(self, custom_path: Optional[str] = None) -> Path:
        """
        Get the vector store path.
        
        Args:
            custom_path (Optional[str]): Custom vector store path
            
        Returns:
            Path: Path to vector store directory
        """
        if custom_path:
            return Path(custom_path)
        return self.vector_store_path
    
    def cleanup_temp_files(self) -> None:
        """Remove all temporary files."""
        try:
            if self.temp_path.exists():
                for temp_file in self.temp_path.glob("*.pdf"):
                    try:
                        temp_file.unlink()
                        logger.debug(f"Removed temp file: {temp_file}")
                    except Exception as e:
                        logger.warning(f"Could not remove temp file {temp_file}: {e}")
        except Exception as e:
            logger.error(f"Error during temp cleanup: {e}")
    
    def validate_path(self, path: Path, must_exist: bool = False) -> bool:
        """
        Validate a path for safety and existence.
        
        Args:
            path (Path): Path to validate
            must_exist (bool): Whether the path must exist
            
        Returns:
            bool: True if path is valid
        """
        try:
            # Check if path is within allowed directories
            path = path.resolve()
            
            # Basic safety checks
            if str(path).startswith('/etc') or str(path).startswith('/sys'):
                return False
            
            if must_exist and not path.exists():
                return False
            
            # Check if parent directory is writable
            if not path.exists():
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Path validation error for {path}: {e}")
            return False 