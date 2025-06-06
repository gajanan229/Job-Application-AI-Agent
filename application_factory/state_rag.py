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
    
    # Phase 4: LLM Integration Fields
    llm_manager_initialized: bool
    ai_generated_resume_sections: Dict[str, str]
    ai_generated_cover_letter: Dict[str, Any]
    extracted_job_skills: Dict[str, List[str]]
    content_enhancement_applied: bool
    ai_generation_timestamp: str


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
    
    # Phase 4: LLM Integration status and content
    llm_manager_initialized=False,
    ai_generated_resume_sections={},
    ai_generated_cover_letter={},
    extracted_job_skills={},
    content_enhancement_applied=False,
    ai_generation_timestamp="",
        
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


def update_vector_store_path(state: GraphStateRAG, vector_store_path: str) -> GraphStateRAG:
    """
    Update the vector store path in the state.
    
    Args:
        state (GraphStateRAG): Current state
        vector_store_path (str): Path to the vector store
        
    Returns:
        GraphStateRAG: Updated state
    """
    updated_state = state.copy()
    updated_state["vector_store_path"] = vector_store_path
    
    # Update processing metadata
    updated_state["processing_metadata"]["vector_store_created"] = True
    updated_state["processing_metadata"]["vector_store_path"] = vector_store_path
    
    logger.info(f"Vector store path updated: {vector_store_path}")
    return updated_state


def update_retrieved_contexts(state: GraphStateRAG, section: str, contexts: List[str]) -> GraphStateRAG:
    """
    Update retrieved contexts for a specific section.
    
    Args:
        state (GraphStateRAG): Current state
        section (str): The section name (e.g., 'summary_context', 'skills_context')
        contexts (List[str]): List of retrieved text chunks
        
    Returns:
        GraphStateRAG: Updated state
    """
    updated_state = state.copy()
    updated_state["retrieved_contexts"][section] = contexts
    
    logger.debug(f"Updated contexts for {section}: {len(contexts)} chunks")
    return updated_state


def update_resume_section(state: GraphStateRAG, section: str, content: str) -> GraphStateRAG:
    """
    Update a specific resume section.
    
    Args:
        state (GraphStateRAG): Current state
        section (str): The section name (e.g., 'summary', 'skills')
        content (str): The generated content
        
    Returns:
        GraphStateRAG: Updated state
    """
    updated_state = state.copy()
    updated_state["resume_sections"][section] = content
    
    # Update processing metadata
    completed_sections = len([s for s in updated_state["resume_sections"].values() if s])
    updated_state["processing_metadata"][f"resume_{section}_completed"] = True
    updated_state["processing_metadata"]["resume_completion_progress"] = f"{completed_sections}/5"
    
    logger.info(f"Resume section '{section}' updated (progress: {completed_sections}/5)")
    return updated_state


def validate_vector_store_in_state(state: GraphStateRAG) -> bool:
    """
    Validate that the vector store referenced in state exists and is valid.
    
    Args:
        state (GraphStateRAG): Current state
        
    Returns:
        bool: True if vector store is valid
    """
    try:
        from pathlib import Path
        
        vector_store_path = state.get("vector_store_path", "")
        if not vector_store_path:
            return False
        
        path = Path(vector_store_path)
        if not path.exists():
            return False
        
        # Check for required FAISS files
        required_files = ["index.faiss", "index.pkl"]
        for file_name in required_files:
            if not (path / file_name).exists():
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating vector store in state: {e}")
        return False


def get_rag_status(state: GraphStateRAG) -> Dict[str, Any]:
    """
    Get detailed RAG status information.
    
    Args:
        state (GraphStateRAG): Current state
        
    Returns:
        Dict[str, Any]: RAG status information
    """
    contexts = state.get("retrieved_contexts", {})
    
    return {
        "vector_store_exists": validate_vector_store_in_state(state),
        "vector_store_path": state.get("vector_store_path", ""),
        "contexts_retrieved": {
            section: len(chunks) for section, chunks in contexts.items()
        },
        "total_contexts": sum(len(chunks) for chunks in contexts.values()),
        "ready_for_generation": bool(state.get("vector_store_path", "") and state.get("master_resume_content", ""))
    }


def update_pdf_paths(state: GraphStateRAG, resume_path: str = "", cover_letter_path: str = "") -> GraphStateRAG:
    """
    Update PDF file paths in the state.
    
    Args:
        state (GraphStateRAG): Current state
        resume_path (str): Path to generated resume PDF
        cover_letter_path (str): Path to generated cover letter PDF
        
    Returns:
        GraphStateRAG: Updated state
    """
    updated_state = state.copy()
    
    if resume_path:
        updated_state["generated_resume_pdf_path"] = resume_path
        updated_state["processing_metadata"]["resume_pdf_generated"] = True
        logger.info(f"Resume PDF path updated: {resume_path}")
    
    if cover_letter_path:
        updated_state["generated_cover_letter_pdf_path"] = cover_letter_path
        updated_state["processing_metadata"]["cover_letter_pdf_generated"] = True
        logger.info(f"Cover letter PDF path updated: {cover_letter_path}")
    
    return updated_state


def get_pdf_status(state: GraphStateRAG) -> Dict[str, Any]:
    """
    Get PDF generation status information.
    
    Args:
        state (GraphStateRAG): Current state
        
    Returns:
        Dict[str, Any]: PDF status information
    """
    from pathlib import Path
    
    resume_path = state.get("generated_resume_pdf_path", "")
    cover_letter_path = state.get("generated_cover_letter_pdf_path", "")
    
    status = {
        "resume_pdf_path": resume_path,
        "cover_letter_pdf_path": cover_letter_path,
        "resume_pdf_exists": bool(resume_path and Path(resume_path).exists()),
        "cover_letter_pdf_exists": bool(cover_letter_path and Path(cover_letter_path).exists()),
        "job_folder_path": state.get("job_specific_output_folder_path", ""),
        "ready_for_pdf_generation": all([
            state.get("resume_sections", {}).get("summary", ""),
            state.get("resume_sections", {}).get("skills", ""),
            state.get("job_specific_output_folder_path", "")
        ])
    }
    
    # Add file sizes if files exist
    try:
        if status["resume_pdf_exists"]:
            status["resume_pdf_size"] = Path(resume_path).stat().st_size
        if status["cover_letter_pdf_exists"]:
            status["cover_letter_pdf_size"] = Path(cover_letter_path).stat().st_size
    except Exception as e:
        logger.debug(f"Could not get PDF file sizes: {e}")
    
    return status


def validate_state_for_pdf_generation(state: GraphStateRAG) -> Dict[str, Any]:
    """
    Validate that the state is ready for PDF generation.
    
    Args:
        state (GraphStateRAG): Current state
        
    Returns:
        Dict[str, Any]: Validation results with status and messages
    """
    validation = {
        "ready": True,
        "issues": [],
        "warnings": []
    }
    
    # Check resume sections
    resume_sections = state.get("resume_sections", {})
    required_sections = ["summary", "skills"]
    
    for section in required_sections:
        if not resume_sections.get(section, "").strip():
            validation["issues"].append(f"Missing required resume section: {section}")
            validation["ready"] = False
    
    # Check optional but recommended sections
    optional_sections = ["experience", "projects", "education"]
    missing_optional = [s for s in optional_sections if not resume_sections.get(s, "").strip()]
    if missing_optional:
        validation["warnings"].append(f"Missing optional sections: {', '.join(missing_optional)}")
    
    # Check job folder path
    if not state.get("job_specific_output_folder_path", ""):
        validation["issues"].append("No job-specific output folder configured")
        validation["ready"] = False
    
    # Check header text
    if not state.get("resume_header_text", ""):
        validation["warnings"].append("No resume header text configured")
    
    return validation


# Phase 4: LLM Integration State Management Functions

def update_llm_initialization_status(state: GraphStateRAG, initialized: bool) -> GraphStateRAG:
    """
    Update the LLM manager initialization status.
    
    Args:
        state (GraphStateRAG): Current state
        initialized (bool): Whether LLM manager is initialized
        
    Returns:
        GraphStateRAG: Updated state
    """
    updated_state = state.copy()
    updated_state["llm_manager_initialized"] = initialized
    
    if initialized:
        import time
        updated_state["ai_generation_timestamp"] = str(time.time())
        logger.info("LLM manager initialized in state")
    else:
        logger.info("LLM manager deinitialized in state")
    
    return updated_state


def update_ai_generated_resume_sections(state: GraphStateRAG, sections: Dict[str, str]) -> GraphStateRAG:
    """
    Update AI-generated resume sections in state.
    
    Args:
        state (GraphStateRAG): Current state
        sections (Dict[str, str]): Generated resume sections
        
    Returns:
        GraphStateRAG: Updated state
    """
    updated_state = state.copy()
    updated_state["ai_generated_resume_sections"] = sections.copy()
    
    # Also update the regular resume_sections for compatibility
    updated_state["resume_sections"].update(sections)
    
    logger.info(f"Updated AI-generated resume sections: {list(sections.keys())}")
    return updated_state


def update_ai_generated_cover_letter(state: GraphStateRAG, cover_letter_data: Dict[str, Any]) -> GraphStateRAG:
    """
    Update AI-generated cover letter in state.
    
    Args:
        state (GraphStateRAG): Current state
        cover_letter_data (Dict[str, Any]): Generated cover letter components
        
    Returns:
        GraphStateRAG: Updated state
    """
    updated_state = state.copy()
    updated_state["ai_generated_cover_letter"] = cover_letter_data.copy()
    
    # Update individual components for compatibility
    if "introduction" in cover_letter_data:
        updated_state["cover_letter_intro"] = cover_letter_data["introduction"]
    
    if "body_paragraphs" in cover_letter_data:
        updated_state["cover_letter_body_paragraphs"] = cover_letter_data["body_paragraphs"]
    
    if "conclusion" in cover_letter_data:
        updated_state["cover_letter_conclusion"] = cover_letter_data["conclusion"]
    
    logger.info("Updated AI-generated cover letter in state")
    return updated_state


def update_extracted_job_skills(state: GraphStateRAG, skills_data: Dict[str, List[str]]) -> GraphStateRAG:
    """
    Update extracted job skills in state.
    
    Args:
        state (GraphStateRAG): Current state
        skills_data (Dict[str, List[str]]): Extracted and categorized skills
        
    Returns:
        GraphStateRAG: Updated state
    """
    updated_state = state.copy()
    updated_state["extracted_job_skills"] = skills_data.copy()
    
    total_skills = sum(len(skills) for skills in skills_data.values())
    logger.info(f"Updated extracted job skills: {total_skills} skills in {len(skills_data)} categories")
    return updated_state


def update_content_enhancement_status(state: GraphStateRAG, applied: bool) -> GraphStateRAG:
    """
    Update content enhancement application status.
    
    Args:
        state (GraphStateRAG): Current state
        applied (bool): Whether content enhancement has been applied
        
    Returns:
        GraphStateRAG: Updated state
    """
    updated_state = state.copy()
    updated_state["content_enhancement_applied"] = applied
    
    if applied:
        logger.info("Content enhancement applied and marked in state")
    
    return updated_state


def get_llm_status(state: GraphStateRAG) -> Dict[str, Any]:
    """
    Get the current LLM integration status from state.
    
    Args:
        state (GraphStateRAG): Current state
        
    Returns:
        Dict[str, Any]: LLM status information
    """
    return {
        "llm_manager_initialized": state.get("llm_manager_initialized", False),
        "ai_generated_sections_count": len(state.get("ai_generated_resume_sections", {})),
        "ai_cover_letter_ready": bool(state.get("ai_generated_cover_letter", {})),
        "job_skills_extracted": bool(state.get("extracted_job_skills", {})),
        "content_enhancement_applied": state.get("content_enhancement_applied", False),
        "ai_generation_timestamp": state.get("ai_generation_timestamp", ""),
        "available_sections": list(state.get("ai_generated_resume_sections", {}).keys()),
        "extracted_skills_categories": list(state.get("extracted_job_skills", {}).keys())
    }


def validate_state_for_llm_generation(state: GraphStateRAG) -> Dict[str, Any]:
    """
    Validate that state is ready for LLM content generation.
    
    Args:
        state (GraphStateRAG): Current state
        
    Returns:
        Dict[str, Any]: Validation results
    """
    validation_result = {
        "ready": True,
        "issues": [],
        "warnings": []
    }
    
    # Check required inputs
    if not state.get("master_resume_content", "").strip():
        validation_result["issues"].append("Master resume content is required")
        validation_result["ready"] = False
    
    if not state.get("job_description_content", "").strip():
        validation_result["issues"].append("Job description is required")
        validation_result["ready"] = False
    
    # Check LLM manager initialization
    if not state.get("llm_manager_initialized", False):
        validation_result["warnings"].append("LLM manager not initialized - will be initialized during generation")
    
    # Check vector store (for RAG enhancement)
    if not state.get("vector_store_path", ""):
        validation_result["warnings"].append("Vector store not available - generation will proceed without RAG enhancement")
    
    # Success message
    if validation_result["ready"]:
        logger.info("State validation passed for LLM generation")
    else:
        logger.error(f"State validation failed for LLM generation: {validation_result['issues']}")
    
    return validation_result 