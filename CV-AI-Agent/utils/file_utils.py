"""File utility functions for the Application Factory."""

import os
import tempfile
import shutil
import time
from pathlib import Path
from typing import Optional, List
import streamlit as st
from datetime import datetime, timedelta

from config.settings import config
from config.logging_config import get_logger

logger = get_logger(__name__)


class FileManager:
    """Utility class for file management operations."""
    
    @staticmethod
    def validate_pdf(uploaded_file) -> tuple[bool, Optional[str]]:
        """
        Validate uploaded PDF file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file size
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > config.max_file_size_mb:
                return False, f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({config.max_file_size_mb}MB)"
            
            # Check file extension
            if not uploaded_file.name.lower().endswith('.pdf'):
                return False, "File must be a PDF document"
            
            # Check if file is empty
            if uploaded_file.size == 0:
                return False, "File appears to be empty"
            
            # Basic PDF header check
            uploaded_file.seek(0)
            header = uploaded_file.read(4)
            uploaded_file.seek(0)  # Reset position
            
            if header != b'%PDF':
                return False, "File does not appear to be a valid PDF document"
            
            logger.debug(f"PDF validation successful for {uploaded_file.name}")
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating PDF file: {e}")
            return False, f"Error validating file: {str(e)}"
    
    @staticmethod
    def save_uploaded_file(uploaded_file, suffix: str = "") -> str:
        """
        Save Streamlit uploaded file to temporary location.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            suffix: Optional suffix to add to filename
        
        Returns:
            Path to saved file
        """
        try:
            # Ensure temp directory exists
            temp_dir = Path(config.temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_name = Path(uploaded_file.name)
            base_name = original_name.stem
            extension = original_name.suffix
            
            if suffix:
                filename = f"{base_name}_{suffix}_{timestamp}{extension}"
            else:
                filename = f"{base_name}_{timestamp}{extension}"
            
            file_path = temp_dir / filename
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            logger.info(f"Saved uploaded file to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            raise
    
    @staticmethod
    def save_text_content(content: str, filename: str) -> str:
        """
        Save text content to a file.
        
        Args:
            content: Text content to save
            filename: Name for the file
            
        Returns:
            Path to the saved file
        """
        try:
            output_dir = Path(config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Sanitize filename
            safe_filename = FileManager._sanitize_filename(filename)
            file_path = output_dir / safe_filename
            
            # Save content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Text content saved to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving text content: {e}")
            raise
    
    @staticmethod
    def cleanup_temp_files(older_than_hours: int = 24) -> int:
        """
        Clean up old temporary files.
        
        Args:
            older_than_hours: Remove files older than this many hours
        
        Returns:
            Number of files removed
        """
        try:
            temp_dir = Path(config.temp_dir)
            if not temp_dir.exists():
                return 0
            
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
            files_removed = 0
            
            for file_path in temp_dir.iterdir():
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        try:
                            file_path.unlink()
                            files_removed += 1
                            logger.debug(f"Removed old temp file: {file_path}")
                        except Exception as e:
                            logger.warning(f"Could not remove temp file {file_path}: {e}")
            
            if files_removed > 0:
                logger.info(f"Cleaned up {files_removed} old temporary files")
            
            return files_removed
            
        except Exception as e:
            logger.error(f"Error during temp file cleanup: {e}")
            return 0
    
    @staticmethod
    def create_output_directory(job_title: str = "", company_name: str = "") -> str:
        """
        Create organized output directory for final documents.
        
        Args:
            job_title: Job title for directory naming
            company_name: Company name for directory naming
        
        Returns:
            Path to created directory
        """
        try:
            base_dir = Path(config.output_dir)
            base_dir.mkdir(parents=True, exist_ok=True)
            
            # Create descriptive folder name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            folder_parts = []
            if company_name:
                # Sanitize company name for filename
                clean_company = FileManager._sanitize_filename(company_name)
                folder_parts.append(clean_company)
            
            if job_title:
                # Sanitize job title for filename
                clean_title = FileManager._sanitize_filename(job_title)
                folder_parts.append(clean_title)
            
            folder_parts.append(timestamp)
            
            if not folder_parts[:-1]:  # Only timestamp
                folder_name = f"Application_{timestamp}"
            else:
                folder_name = "_".join(folder_parts)
            
            output_dir = base_dir / folder_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Created output directory: {output_dir}")
            return str(output_dir)
            
        except Exception as e:
            logger.error(f"Error creating output directory: {e}")
            raise
    
    @staticmethod
    def _sanitize_filename(filename: str, max_length: int = 50) -> str:
        """
        Sanitize string for use in filename.
        
        Args:
            filename: String to sanitize
            max_length: Maximum length of sanitized string
        
        Returns:
            Sanitized filename string
        """
        # Replace problematic characters
        import re
        
        # Remove/replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Replace multiple spaces with single underscore
        sanitized = re.sub(r'\s+', '_', sanitized)
        
        # Remove leading/trailing underscores and dots
        sanitized = sanitized.strip('_.')
        
        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length].strip('_.')
        
        # Ensure not empty
        if not sanitized:
            sanitized = "Document"
        
        return sanitized
    
    @staticmethod
    def get_file_info(file_path: str) -> dict:
        """
        Get information about a file.
        
        Args:
            file_path: Path to file
        
        Returns:
            Dictionary with file information
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return {"exists": False}
            
            stat = path.stat()
            return {
                "exists": True,
                "size_bytes": stat.st_size,
                "size_mb": stat.st_size / (1024 * 1024),
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "extension": path.suffix,
                "name": path.name,
                "stem": path.stem
            }
            
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {"exists": False, "error": str(e)}
    
    @staticmethod
    def ensure_file_exists(file_path: str) -> bool:
        """
        Check if a file exists and is accessible.
        
        Args:
            file_path: Path to check
        
        Returns:
            True if file exists and is accessible
        """
        try:
            path = Path(file_path)
            return path.exists() and path.is_file()
        except Exception:
            return False
    
    @staticmethod
    def copy_file(source: str, destination: str) -> bool:
        """
        Copy a file from source to destination.
        
        Args:
            source: Source file path
            destination: Destination file path
        
        Returns:
            True if copy was successful
        """
        try:
            # Ensure destination directory exists
            dest_path = Path(destination)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(source, destination)
            logger.debug(f"Copied file from {source} to {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Error copying file from {source} to {destination}: {e}")
            return False
    
    @staticmethod
    def get_temp_file_path(prefix: str = "temp", suffix: str = ".tmp") -> str:
        """
        Generate a temporary file path.
        
        Args:
            prefix: Prefix for temp file name
            suffix: File extension/suffix
        
        Returns:
            Path to temporary file
        """
        temp_dir = Path(config.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}{suffix}"
        
        return str(temp_dir / filename)


def ensure_directory_exists(directory_path: Path) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path object for the directory to ensure exists
    """
    try:
        directory_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory_path}")
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {e}")
        raise