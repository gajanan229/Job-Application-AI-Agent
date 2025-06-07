"""Session state utilities for the Application Factory."""

import streamlit as st
from typing import Any, Dict, Optional, List
from datetime import datetime

from config.logging_config import get_logger

logger = get_logger(__name__)


class SessionManager:
    """Utility class for managing Streamlit session state."""
    
    # Session state keys
    APP_STATE_KEY = "app_state"
    PROCESSING_STATUS_KEY = "processing_status"
    ERROR_MESSAGES_KEY = "error_messages"
    SUCCESS_MESSAGES_KEY = "success_messages"
    TEMP_FILES_KEY = "temp_files"
    CURRENT_STAGE_KEY = "current_stage"
    
    @staticmethod
    def initialize_session():
        """Initialize session state with default values."""
        try:
            # Initialize app state if not exists
            if SessionManager.APP_STATE_KEY not in st.session_state:
                st.session_state[SessionManager.APP_STATE_KEY] = {
                    # File paths
                    "master_resume_pdf_path": None,
                    "job_description_pdf_path": None,
                    "master_resume_text": None,
                    "job_description_text": None,
                    "master_resume_vectorstore": None,
                    "jd_vectorstore": None,
                    
                    # Resume specific
                    "generated_summary": None,
                    "edited_summary": None,
                    "generated_projects": None,
                    "edited_projects": None,
                    "current_resume_docx": None,
                    "current_resume_docx_path": None,
                    "current_resume_pdf_path": None,
                    
                    # Cover Letter specific
                    "cl_intro": None,
                    "cl_body": None,
                    "cl_conclusion": None,
                    "cl_intro_versions": [],
                    "cl_body_versions": [],
                    "cl_conclusion_versions": [],
                    "user_notes_cl_intro": None,
                    "user_notes_cl_body": None,
                    "user_notes_cl_conclusion": None,
                    "current_cl_docx": None,
                    "current_cl_docx_path": None,
                    "current_cl_pdf_path": None,
                    
                    # Final paths for download
                    "final_resume_docx_path": None,
                    "final_resume_pdf_path": None,
                    "final_cl_docx_path": None,
                    "final_cl_pdf_path": None,
                    
                    # Progress and Status
                    "current_stage": "setup",
                    "processing_status": None,
                    "error_message": None,
                    
                    # File Management
                    "temp_files": [],
                    
                    # User Preferences
                    "company_name": None,
                    "job_title": None,
                    "user_name": None,
                    
                    # Control flow
                    "next_node": None,
                    "resume_edit_triggered": False,
                    "cl_regeneration_triggered": False,
                    
                    # Workflow Engine State
                    "workflow_engine": None,
                    "workflow_state": None,
                    "workflow_thread_id": None,
                    "workflow_progress": None
                }
            
            # Initialize other session state keys
            if SessionManager.PROCESSING_STATUS_KEY not in st.session_state:
                st.session_state[SessionManager.PROCESSING_STATUS_KEY] = None
                
            if SessionManager.ERROR_MESSAGES_KEY not in st.session_state:
                st.session_state[SessionManager.ERROR_MESSAGES_KEY] = []
                
            if SessionManager.SUCCESS_MESSAGES_KEY not in st.session_state:
                st.session_state[SessionManager.SUCCESS_MESSAGES_KEY] = []
                
            if SessionManager.TEMP_FILES_KEY not in st.session_state:
                st.session_state[SessionManager.TEMP_FILES_KEY] = []
                
            if SessionManager.CURRENT_STAGE_KEY not in st.session_state:
                st.session_state[SessionManager.CURRENT_STAGE_KEY] = "setup"
            
            logger.debug("Session state initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing session state: {e}")
            raise
    
    @staticmethod
    def update_app_state(updates: Dict[str, Any]):
        """
        Safely update app state in session.
        
        Args:
            updates: Dictionary of updates to apply to app state
        """
        try:
            if SessionManager.APP_STATE_KEY not in st.session_state:
                SessionManager.initialize_session()
            
            # Update app state
            current_state = st.session_state[SessionManager.APP_STATE_KEY]
            current_state.update(updates)
            st.session_state[SessionManager.APP_STATE_KEY] = current_state
            
            logger.debug(f"Updated app state with keys: {list(updates.keys())}")
            
        except Exception as e:
            logger.error(f"Error updating app state: {e}")
            raise
    
    @staticmethod
    def get_app_state() -> Dict[str, Any]:
        """
        Get current app state from session.
        
        Returns:
            Current app state dictionary
        """
        try:
            if SessionManager.APP_STATE_KEY not in st.session_state:
                SessionManager.initialize_session()
            
            return st.session_state[SessionManager.APP_STATE_KEY].copy()
            
        except Exception as e:
            logger.error(f"Error getting app state: {e}")
            return {}
    
    @staticmethod
    def get_app_state_value(key: str, default: Any = None) -> Any:
        """
        Get a specific value from app state.
        
        Args:
            key: Key to retrieve
            default: Default value if key not found
        
        Returns:
            Value from app state or default
        """
        try:
            app_state = SessionManager.get_app_state()
            return app_state.get(key, default)
            
        except Exception as e:
            logger.error(f"Error getting app state value for key '{key}': {e}")
            return default
    
    @staticmethod
    def set_app_state_value(key: str, value: Any):
        """
        Set a specific value in app state.
        
        Args:
            key: Key to set
            value: Value to set
        """
        SessionManager.update_app_state({key: value})
    
    @staticmethod
    def clear_session():
        """Clear all session state."""
        try:
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # Reinitialize with defaults
            SessionManager.initialize_session()
            
            logger.info("Session state cleared and reinitialized")
            
        except Exception as e:
            logger.error(f"Error clearing session state: {e}")
            raise
    
    @staticmethod
    def set_processing_status(status: Optional[str]):
        """
        Set processing status message.
        
        Args:
            status: Status message to display
        """
        st.session_state[SessionManager.PROCESSING_STATUS_KEY] = status
        SessionManager.set_app_state_value("processing_status", status)
        
        if status:
            logger.info(f"Processing status: {status}")
    
    @staticmethod
    def get_processing_status() -> Optional[str]:
        """Get current processing status."""
        return st.session_state.get(SessionManager.PROCESSING_STATUS_KEY)
    
    @staticmethod
    def add_error_message(message: str):
        """
        Add an error message to display.
        
        Args:
            message: Error message
        """
        if SessionManager.ERROR_MESSAGES_KEY not in st.session_state:
            st.session_state[SessionManager.ERROR_MESSAGES_KEY] = []
        
        st.session_state[SessionManager.ERROR_MESSAGES_KEY].append({
            "message": message,
            "timestamp": datetime.now()
        })
        
        SessionManager.set_app_state_value("error_message", message)
        logger.error(f"Added error message: {message}")
    
    @staticmethod
    def add_success_message(message: str):
        """
        Add a success message to display.
        
        Args:
            message: Success message
        """
        if SessionManager.SUCCESS_MESSAGES_KEY not in st.session_state:
            st.session_state[SessionManager.SUCCESS_MESSAGES_KEY] = []
        
        st.session_state[SessionManager.SUCCESS_MESSAGES_KEY].append({
            "message": message,
            "timestamp": datetime.now()
        })
        
        logger.info(f"Added success message: {message}")
    
    @staticmethod
    def get_error_messages() -> List[Dict[str, Any]]:
        """Get all error messages."""
        return st.session_state.get(SessionManager.ERROR_MESSAGES_KEY, [])
    
    @staticmethod
    def get_success_messages() -> List[Dict[str, Any]]:
        """Get all success messages."""
        return st.session_state.get(SessionManager.SUCCESS_MESSAGES_KEY, [])
    
    @staticmethod
    def clear_messages():
        """Clear all messages."""
        st.session_state[SessionManager.ERROR_MESSAGES_KEY] = []
        st.session_state[SessionManager.SUCCESS_MESSAGES_KEY] = []
        SessionManager.set_app_state_value("error_message", None)
    
    @staticmethod
    def add_temp_file(file_path: str):
        """
        Add a temporary file to track for cleanup.
        
        Args:
            file_path: Path to temporary file
        """
        if SessionManager.TEMP_FILES_KEY not in st.session_state:
            st.session_state[SessionManager.TEMP_FILES_KEY] = []
        
        if file_path not in st.session_state[SessionManager.TEMP_FILES_KEY]:
            st.session_state[SessionManager.TEMP_FILES_KEY].append(file_path)
        
        # Also update app state
        temp_files = SessionManager.get_app_state_value("temp_files", [])
        if file_path not in temp_files:
            temp_files.append(file_path)
            SessionManager.set_app_state_value("temp_files", temp_files)
    
    @staticmethod
    def get_temp_files() -> List[str]:
        """Get list of temporary files."""
        return st.session_state.get(SessionManager.TEMP_FILES_KEY, [])
    
    @staticmethod
    def set_current_stage(stage: str):
        """
        Set the current application stage.
        
        Args:
            stage: Stage identifier
        """
        st.session_state[SessionManager.CURRENT_STAGE_KEY] = stage
        SessionManager.set_app_state_value("current_stage", stage)
        logger.info(f"Set current stage: {stage}")
    
    @staticmethod
    def get_current_stage() -> str:
        """Get current application stage."""
        return st.session_state.get(SessionManager.CURRENT_STAGE_KEY, "setup")
    
    @staticmethod
    def is_stage(stage: str) -> bool:
        """
        Check if current stage matches given stage.
        
        Args:
            stage: Stage to check
        
        Returns:
            True if current stage matches
        """
        return SessionManager.get_current_stage() == stage
    
    @staticmethod
    def save_cover_letter_version(section: str, content: str):
        """
        Save a version of a cover letter section.
        
        Args:
            section: Section name ('intro', 'body', 'conclusion')
            content: Content to save
        """
        versions_key = f"cl_{section}_versions"
        current_versions = SessionManager.get_app_state_value(versions_key, [])
        
        # Add current content if not empty and different from last version
        if content and (not current_versions or current_versions[-1] != content):
            current_versions.append(content)
            
            # Keep only last 10 versions to avoid memory issues
            if len(current_versions) > 10:
                current_versions = current_versions[-10:]
            
            SessionManager.set_app_state_value(versions_key, current_versions)
            logger.debug(f"Saved version for {section}, total versions: {len(current_versions)}")
    
    @staticmethod
    def get_cover_letter_versions(section: str) -> List[str]:
        """
        Get versions of a cover letter section.
        
        Args:
            section: Section name ('intro', 'body', 'conclusion')
        
        Returns:
            List of previous versions
        """
        versions_key = f"cl_{section}_versions"
        return SessionManager.get_app_state_value(versions_key, [])
    
    @staticmethod
    def revert_cover_letter_version(section: str) -> Optional[str]:
        """
        Revert to previous version of a cover letter section.
        
        Args:
            section: Section name ('intro', 'body', 'conclusion')
        
        Returns:
            Previous version content or None if no previous version
        """
        versions = SessionManager.get_cover_letter_versions(section)
        
        if len(versions) > 1:
            # Remove current version and return previous
            versions.pop()
            SessionManager.set_app_state_value(f"cl_{section}_versions", versions)
            
            previous_content = versions[-1] if versions else None
            if previous_content:
                SessionManager.set_app_state_value(f"cl_{section}", previous_content)
                logger.info(f"Reverted {section} to previous version")
            
            return previous_content
        
        return None
    
    @staticmethod
    def get_timestamp() -> str:
        """
        Get current timestamp string for file naming.
        
        Returns:
            Timestamp string in format YYYYMMDD_HHMMSS
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def get_session_info() -> Dict[str, Any]:
        """
        Get information about current session state.
        
        Returns:
            Dictionary with session information
        """
        return {
            "current_stage": SessionManager.get_current_stage(),
            "processing_status": SessionManager.get_processing_status(),
            "error_count": len(SessionManager.get_error_messages()),
            "success_count": len(SessionManager.get_success_messages()),
            "temp_files_count": len(SessionManager.get_temp_files()),
            "has_master_resume": SessionManager.get_app_state_value("master_resume_pdf_path") is not None,
            "has_job_description": SessionManager.get_app_state_value("job_description_pdf_path") is not None,
            "has_generated_resume": SessionManager.get_app_state_value("current_resume_pdf_path") is not None,
            "has_generated_cover_letter": SessionManager.get_app_state_value("current_cl_pdf_path") is not None,
            "workflow_state": SessionManager.get_app_state_value("workflow_state"),
            "workflow_progress": SessionManager.get_app_state_value("workflow_progress")
        }
    
    @staticmethod
    def initialize_workflow_engine():
        """
        Initialize the LangGraph workflow engine.
        """
        try:
            from core.graph import ApplicationFactoryWorkflow
            
            workflow_engine = ApplicationFactoryWorkflow()
            SessionManager.set_app_state_value("workflow_engine", workflow_engine)
            
            logger.info("Workflow engine initialized successfully")
            return workflow_engine
            
        except Exception as e:
            logger.error(f"Error initializing workflow engine: {e}")
            raise
    
    @staticmethod
    def get_workflow_engine():
        """Get the current workflow engine instance."""
        return SessionManager.get_app_state_value("workflow_engine")
    
    @staticmethod
    def set_workflow_state(state: Dict[str, Any]):
        """Set the workflow state."""
        SessionManager.set_app_state_value("workflow_state", state)
    
    @staticmethod
    def get_workflow_state() -> Optional[Dict[str, Any]]:
        """Get the current workflow state."""
        return SessionManager.get_app_state_value("workflow_state")
    
    @staticmethod
    def set_workflow_thread_id(thread_id: str):
        """Set the workflow thread ID."""
        SessionManager.set_app_state_value("workflow_thread_id", thread_id)
    
    @staticmethod
    def get_workflow_thread_id() -> Optional[str]:
        """Get the workflow thread ID."""
        return SessionManager.get_app_state_value("workflow_thread_id")
    
    @staticmethod
    def update_workflow_progress(progress: Dict[str, Any]):
        """Update workflow progress information."""
        SessionManager.set_app_state_value("workflow_progress", progress) 