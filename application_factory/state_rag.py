"""
State management for the Application Factory LangGraph workflow.

Defines the GraphStateRAG TypedDict that contains all necessary state
for the RAG-enhanced resume and cover letter generation process.
"""

from typing import TypedDict, Dict, List, Optional, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class GraphStateRAG(TypedDict):
    """
    State container for the LangGraph workflow.
    
    This TypedDict contains all the state needed for the Application Factory
    workflow, including file paths, content, generated sections, and metadata.
    
    NOTE: API keys are NOT stored in state for security reasons.
    Use config.api_keys.get_gemini_api_key() when needed.
    """
    
    # Input File Paths and Content
    master_resume_path: str
    master_resume_content: str
    job_description_content: str
    
    # Vector Store Configuration
    vector_store_path: str
    
    # RAG Retrieved Contexts (organized by purpose)
    retrieved_contexts: Dict[str, List[str]]
    
    # Generated Resume Sections
    resume_sections: Dict[str, str]
    
    # Generated Resume PDF
    generated_resume_pdf_path: str
    
    # Cover Letter Components
    cover_letter_intro: str
    cover_letter_conclusion: str
    cover_letter_body_paragraphs: List[str]
    
    # Cover Letter Version Management
    cover_letter_versions: Dict[str, List[Dict[str, Any]]]
    
    # User Feedback for Iterative Improvement
    user_feedback_cl: str
    
    # Generated Cover Letter PDF
    generated_cover_letter_pdf_path: str
    
    # Output Organization
    job_specific_output_folder_path: str
    
    # Error Handling
    error_message: Optional[str]
    
    # Workflow Management
    current_stage: str
    
    # Header Information for PDFs
    resume_header_text: Optional[str]
    cover_letter_header_text: Optional[str]
    
    # Processing Metadata
    processing_metadata: Dict[str, Any]


def create_initial_state(
    master_resume_path: str = "",
    job_description_content: str = "",
    output_base_path: str = "generated_applications",
    resume_header: str = "",
    cover_letter_header: str = ""
) -> GraphStateRAG:
    """
    Create an initial state for the LangGraph workflow.
    
    Args:
        master_resume_path (str): Path to the master resume file
        job_description_content (str): Job description text
        output_base_path (str): Base path for output files
        resume_header (str): Header text for resume
        cover_letter_header (str): Header text for cover letter
        
    Returns:
        GraphStateRAG: Initial state dictionary
    """
    return GraphStateRAG(
        # Input paths and content
        master_resume_path=master_resume_path,
        master_resume_content="",
        job_description_content=job_description_content,
        
        # Vector store
        vector_store_path="",
        
        # RAG contexts (will be populated during workflow)
        retrieved_contexts={
            "summary_context": [],
            "skills_context": [],
            "education_context": [],
            "experience_context": [],
            "projects_context": [],
            "cl_ic_context": [],  # cover letter intro/conclusion context
            "cl_body_context": []  # cover letter body context
        },
        
        # Resume sections (will be generated)
        resume_sections={
            "summary": "",
            "skills": "",
            "education": "",
            "experience": "",
            "projects": ""
        },
        
        # Generated files
        generated_resume_pdf_path="",
        generated_cover_letter_pdf_path="",
        
        # Cover letter components
        cover_letter_intro="",
        cover_letter_conclusion="",
        cover_letter_body_paragraphs=[],
        
        # Version management for iterative improvement
        cover_letter_versions={
            "intro": [],
            "conclusion": [],
            "body": []
        },
        
        # User feedback
        user_feedback_cl="",
        
        # Output organization
        job_specific_output_folder_path="",
        
        # Error handling
        error_message=None,
        
        # Workflow state
        current_stage="initialization",
        
        # Headers
        resume_header_text=resume_header,
        cover_letter_header_text=cover_letter_header,
        
        # Processing metadata
        processing_metadata={
            "timestamp_start": "",
            "timestamp_last_update": "",
            "workflow_version": "1.0.0",
            "stages_completed": [],
            "retry_counts": {},
            "performance_metrics": {}
        }
    )


def update_state_stage(state: GraphStateRAG, new_stage: str) -> GraphStateRAG:
    """
    Update the current stage in the state.
    
    Args:
        state (GraphStateRAG): Current state
        new_stage (str): New stage name
        
    Returns:
        GraphStateRAG: Updated state
    """
    import time
    
    updated_state = state.copy()
    updated_state["current_stage"] = new_stage
    updated_state["processing_metadata"]["timestamp_last_update"] = str(time.time())
    
    # Add to completed stages if not already there
    if updated_state["current_stage"] not in updated_state["processing_metadata"]["stages_completed"]:
        updated_state["processing_metadata"]["stages_completed"].append(new_stage)
    
    return updated_state


def set_error_state(state: GraphStateRAG, error_message: str, stage: str = None) -> GraphStateRAG:
    """
    Set an error state with appropriate message.
    
    Args:
        state (GraphStateRAG): Current state
        error_message (str): Error message to set
        stage (str, optional): Stage where error occurred
        
    Returns:
        GraphStateRAG: Updated state with error
    """
    updated_state = state.copy()
    updated_state["error_message"] = error_message
    
    if stage:
        updated_state["current_stage"] = f"error_in_{stage}"
        # Increment retry count for this stage
        retry_key = f"{stage}_retries"
        updated_state["processing_metadata"]["retry_counts"][retry_key] = \
            updated_state["processing_metadata"]["retry_counts"].get(retry_key, 0) + 1
    
    logger.error(f"State error in stage {stage}: {error_message}")
    return updated_state


def clear_error_state(state: GraphStateRAG) -> GraphStateRAG:
    """
    Clear any error state.
    
    Args:
        state (GraphStateRAG): Current state
        
    Returns:
        GraphStateRAG: Updated state with cleared error
    """
    updated_state = state.copy()
    updated_state["error_message"] = None
    return updated_state


def validate_state(state: GraphStateRAG, required_fields: List[str] = None) -> bool:
    """
    Validate that the state contains required fields and is properly formatted.
    
    Args:
        state (GraphStateRAG): State to validate
        required_fields (List[str], optional): Additional required fields to check
        
    Returns:
        bool: True if state is valid
    """
    try:
        # Check basic state structure
        if not isinstance(state, dict):
            logger.error("State is not a dictionary")
            return False
        
        # Check for required base fields
        base_required = [
            "master_resume_content", "job_description_content", 
            "current_stage", "retrieved_contexts", "resume_sections"
        ]
        
        for field in base_required:
            if field not in state:
                logger.error(f"Required field '{field}' missing from state")
                return False
        
        # Check additional required fields if specified
        if required_fields:
            for field in required_fields:
                if field not in state or not state[field]:
                    logger.error(f"Required field '{field}' missing or empty in state")
                    return False
        
        # Validate data types
        if not isinstance(state["retrieved_contexts"], dict):
            logger.error("retrieved_contexts must be a dictionary")
            return False
        
        if not isinstance(state["resume_sections"], dict):
            logger.error("resume_sections must be a dictionary")
            return False
        
        if not isinstance(state["cover_letter_body_paragraphs"], list):
            logger.error("cover_letter_body_paragraphs must be a list")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"State validation error: {e}")
        return False


def serialize_state_for_logging(state: GraphStateRAG) -> str:
    """
    Serialize state for logging purposes (excluding sensitive content).
    
    Args:
        state (GraphStateRAG): State to serialize
        
    Returns:
        str: JSON string of safe state information
    """
    safe_state = {
        "current_stage": state.get("current_stage", "unknown"),
        "has_master_resume": bool(state.get("master_resume_content", "")),
        "has_job_description": bool(state.get("job_description_content", "")),
        "has_vector_store": bool(state.get("vector_store_path", "")),
        "resume_sections_ready": len([s for s in state.get("resume_sections", {}).values() if s]),
        "cover_letter_ready": bool(state.get("cover_letter_intro", "") and state.get("cover_letter_conclusion", "")),
        "error_present": bool(state.get("error_message")),
        "processing_metadata": state.get("processing_metadata", {})
    }
    
    try:
        return json.dumps(safe_state, indent=2)
    except Exception as e:
        logger.error(f"Error serializing state for logging: {e}")
        return str(safe_state)


def get_state_summary(state: GraphStateRAG) -> Dict[str, Any]:
    """
    Get a summary of the current state for UI display.
    
    Args:
        state (GraphStateRAG): Current state
        
    Returns:
        Dict[str, Any]: Summary information
    """
    return {
        "stage": state.get("current_stage", "unknown"),
        "master_resume_loaded": bool(state.get("master_resume_content", "")),
        "job_description_loaded": bool(state.get("job_description_content", "")),
        "vector_store_ready": bool(state.get("vector_store_path", "")),
        "resume_sections_count": len([s for s in state.get("resume_sections", {}).values() if s]),
        "cover_letter_progress": {
            "intro": bool(state.get("cover_letter_intro", "")),
            "conclusion": bool(state.get("cover_letter_conclusion", "")),
            "body_paragraphs": len(state.get("cover_letter_body_paragraphs", []))
        },
        "has_error": bool(state.get("error_message")),
        "error_message": state.get("error_message"),
        "output_folder": state.get("job_specific_output_folder_path", "")
    } 