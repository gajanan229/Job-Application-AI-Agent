"""
Path management utilities for the Application Factory.

Handles creation, validation, and management of file paths and directories.
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
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
        
        # Enhanced extraction patterns
        company_keywords = ['company:', 'organization:', 'employer:', 'corporation:', 'firm:', 'about us:', 'at ']
        position_keywords = ['position:', 'title:', 'role:', 'job:', 'opening:', 'opportunity:', 'seeking:']
        
        # Try to extract company and position from first few lines
        for i, line in enumerate(lines[:15]):  # Check first 15 lines
            line = line.strip()
            line_lower = line.lower()
            
            # Look for company patterns
            for keyword in company_keywords:
                if keyword in line_lower:
                    if ':' in line:
                        company_name = line.split(':', 1)[-1].strip()
                    else:
                        # Extract everything after the keyword
                        idx = line_lower.find(keyword)
                        if idx >= 0:
                            company_name = line[idx + len(keyword):].strip()
                    break
            
            # Look for position patterns
            for keyword in position_keywords:
                if keyword in line_lower:
                    if ':' in line:
                        position_title = line.split(':', 1)[-1].strip()
                    else:
                        # Extract everything after the keyword
                        idx = line_lower.find(keyword)
                        if idx >= 0:
                            position_title = line[idx + len(keyword):].strip()
                    break
            
            # Fallback patterns for first few lines
            if i == 0 and len(line) > 10 and position_title == "Position":
                # First line is often the job title
                position_title = line
            elif i == 1 and len(line) > 5 and company_name == "Company":
                # Second line might be the company
                company_name = line
        
        # Additional cleanup for common formats
        # Remove common prefixes/suffixes
        for prefix in ['job title:', 'position:', 'role:', 'opening for']:
            if position_title.lower().startswith(prefix):
                position_title = position_title[len(prefix):].strip()
        
        for prefix in ['company:', 'organization:', 'employer:']:
            if company_name.lower().startswith(prefix):
                company_name = company_name[len(prefix):].strip()
        
        # Remove trailing punctuation
        position_title = position_title.rstrip('.,;:')
        company_name = company_name.rstrip('.,;:')
        
        # Clean and sanitize
        company_name = self.sanitize_filename(company_name) if company_name else "Company"
        position_title = self.sanitize_filename(position_title) if position_title else "Position"
        
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
    
    def get_resume_preview_path(self, user_name: str = "Resume") -> Path:
        """
        Get the path for a resume preview PDF.
        
        Args:
            user_name (str): User's name for the filename
            
        Returns:
            Path: Path to temporary resume preview PDF
        """
        sanitized_name = self.sanitize_filename(user_name)
        filename = f"{sanitized_name}_Resume_Preview.pdf"
        return self.temp_path / filename
    
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
    
    def get_cover_letter_preview_path(self, user_name: str = "CoverLetter") -> Path:
        """
        Get the path for a cover letter preview PDF.
        
        Args:
            user_name (str): User's name for the filename
            
        Returns:
            Path: Path to temporary cover letter preview PDF
        """
        sanitized_name = self.sanitize_filename(user_name)
        filename = f"{sanitized_name}_CoverLetter_Preview.pdf"
        return self.temp_path / filename
    
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
    
    def get_master_resume_vector_store_path(self) -> Path:
        """
        Get the specific path for master resume vector store.
        
        Returns:
            Path: Path to master resume vector store
        """
        return self.vector_store_path / "master_resume"
    
    def create_vector_store_folder(self, store_name: str) -> Path:
        """
        Create a vector store folder.
        
        Args:
            store_name (str): Name of the vector store
            
        Returns:
            Path: Path to the created vector store folder
        """
        sanitized_name = self.sanitize_filename(store_name)
        store_path = self.vector_store_path / sanitized_name
        
        try:
            store_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created vector store folder: {store_path}")
            return store_path
        except Exception as e:
            logger.error(f"Failed to create vector store folder {store_path}: {e}")
            raise
    
    def validate_vector_store_path(self, store_path: Path) -> bool:
        """
        Validate that a vector store path exists and contains required files.
        
        Args:
            store_path (Path): Path to validate
            
        Returns:
            bool: True if valid vector store path
        """
        if not store_path.exists():
            return False
        
        # Check for required FAISS files
        required_files = ["index.faiss", "index.pkl"]
        for file_name in required_files:
            if not (store_path / file_name).exists():
                return False
        
        return True
    
    def cleanup_vector_stores(self, keep_recent: int = 5) -> None:
        """
        Clean up old vector stores, keeping only the most recent ones.
        
        Args:
            keep_recent (int): Number of recent vector stores to keep
        """
        try:
            if not self.vector_store_path.exists():
                return
            
            # Get all vector store directories
            store_dirs = [d for d in self.vector_store_path.iterdir() if d.is_dir()]
            
            # Sort by modification time (newest first)
            store_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old stores
            for old_store in store_dirs[keep_recent:]:
                try:
                    import shutil
                    shutil.rmtree(old_store)
                    logger.info(f"Removed old vector store: {old_store}")
                except Exception as e:
                    logger.warning(f"Could not remove vector store {old_store}: {e}")
                    
        except Exception as e:
            logger.error(f"Error during vector store cleanup: {e}")
    
    def validate_pdf_path(self, pdf_path: Path) -> bool:
        """
        Validate that a PDF path is safe and accessible.
        
        Args:
            pdf_path (Path): Path to validate
            
        Returns:
            bool: True if valid PDF path
        """
        try:
            # Check if path has PDF extension
            if pdf_path.suffix.lower() != '.pdf':
                return False
            
            # Validate parent directory
            return self.validate_path(pdf_path.parent, must_exist=False)
            
        except Exception as e:
            logger.error(f"Error validating PDF path {pdf_path}: {e}")
            return False
    
    def create_application_package_paths(self, job_description: str, user_name: str) -> Dict[str, Path]:
        """
        Create all necessary paths for a complete application package.
        
        Args:
            job_description (str): Job description to extract company/position
            user_name (str): User's name for file naming
            
        Returns:
            Dict[str, Path]: Dictionary of all application paths
        """
        try:
            # Create job folder
            job_folder = self.create_job_folder(job_description)
            
            # Generate all paths
            paths = {
                "job_folder": job_folder,
                "resume_pdf": self.get_resume_path(job_folder, user_name),
                "cover_letter_pdf": self.get_cover_letter_path(job_folder, user_name),
                "resume_preview": self.get_resume_preview_path(user_name),
                "cover_letter_preview": self.get_cover_letter_preview_path(user_name)
            }
            
            logger.info(f"Created application package paths for job folder: {job_folder}")
            return paths
            
        except Exception as e:
            logger.error(f"Error creating application package paths: {e}")
            return {}
    
    def cleanup_preview_files(self) -> None:
        """Clean up all preview PDF files."""
        try:
            if not self.temp_path.exists():
                return
            
            # Remove all preview PDF files
            for pdf_file in self.temp_path.glob("*_Preview.pdf"):
                try:
                    pdf_file.unlink()
                    logger.debug(f"Removed preview file: {pdf_file}")
                except Exception as e:
                    logger.warning(f"Could not remove preview file {pdf_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error during preview cleanup: {e}")
    
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