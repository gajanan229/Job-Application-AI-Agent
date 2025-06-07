"""
LangGraph Workflow Engine for Application Factory.

This module implements the complete state machine for orchestrating the entire
resume and cover letter generation process using LangGraph.
"""

import os
import operator
from datetime import datetime
from typing import TypedDict, Annotated, Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

from langgraph.graph import StateGraph, START, END

from config.settings import config
from config.logging_config import get_logger, timing_decorator
from utils.session_utils import SessionManager
from utils.error_handlers import error_handler

# Import all our components
from core.rag_processor import RAGProcessor
from core.llm_service import ContentGenerator, ContentType, GenerationResponse
from core.document_generator import DocumentGenerator, DocumentData, GeneratedDocument
from utils.content_validators import ContentValidator, validate_resume, validate_cover_letter

logger = get_logger(__name__)


def merge_dicts(left: Dict, right: Dict) -> Dict:
    """
    Merge two dictionaries for LangGraph state updates.
    
    Args:
        left: Current dictionary state
        right: New dictionary to merge in
        
    Returns:
        Merged dictionary
    """
    if left is None:
        return right or {}
    if right is None:
        return left or {}
    
    # Create a copy of left and update with right
    result = left.copy()
    result.update(right)
    return result


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeStatus(Enum):
    """Individual node execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowProgress:
    """Track workflow execution progress."""
    current_node: str
    completed_nodes: List[str]
    failed_nodes: List[str]
    total_nodes: int
    progress_percentage: float
    status: WorkflowStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        if self.start_time:
            result['start_time'] = self.start_time.isoformat()
        if self.end_time:
            result['end_time'] = self.end_time.isoformat()
        result['status'] = self.status.value
        return result


class AppState(TypedDict):
    """
    Complete application state for the LangGraph workflow.
    
    This TypedDict defines all the state that flows through the workflow,
    including inputs, intermediate results, and final outputs.
    """
    # Input files and configuration
    api_key: str
    master_resume_pdf_path: str
    job_description_pdf_path: str
    
    # RAG Processing Results
    rag_processor: Optional[RAGProcessor]
    master_resume_result: Optional[Dict[str, Any]]
    job_description_result: Optional[Dict[str, Any]]
    
    # Content Generation Results
    content_generator: Optional[ContentGenerator]
    resume_summary: Optional[GenerationResponse]
    resume_projects: Optional[GenerationResponse]
    cover_letter_intro: Optional[GenerationResponse]
    cover_letter_body: Optional[GenerationResponse]
    cover_letter_conclusion: Optional[GenerationResponse]
    
    # Document Generation Results
    document_generator: Optional[DocumentGenerator]
    resume_document: Optional[GeneratedDocument]
    cover_letter_document: Optional[GeneratedDocument]
    
    # Workflow Control and Status
    workflow_progress: WorkflowProgress
    node_statuses: Annotated[Dict[str, NodeStatus], merge_dicts]
    errors: Annotated[List[str], operator.add]
    logs: Annotated[List[str], operator.add]
    
    # Configuration and preferences
    user_preferences: Dict[str, Any]
    session_id: str
    timestamp: str


class ApplicationFactoryWorkflow:
    """
    LangGraph workflow engine for the Application Factory.
    
    This class orchestrates the complete process of transforming a master resume
    and job description into tailored resume and cover letter documents.
    """
    
    def __init__(self):
        """Initialize the workflow engine."""
        self.graph = None
        self.compiled_graph = None
        self._initialize_graph()
    
    def _initialize_graph(self):
        """Build and compile the LangGraph workflow."""
        logger.info("Initializing LangGraph workflow...")
        
        # Create the state graph
        self.graph = StateGraph(AppState)
        
        # Add all workflow nodes
        self.graph.add_node("initialize_workflow", self.initialize_workflow_node)
        self.graph.add_node("process_documents", self.process_documents_node)
        self.graph.add_node("generate_resume_summary", self.generate_resume_summary_node)
        self.graph.add_node("extract_resume_projects", self.extract_resume_projects_node)
        self.graph.add_node("generate_cover_letter_intro", self.generate_cover_letter_intro_node)
        self.graph.add_node("generate_cover_letter_conclusion", self.generate_cover_letter_conclusion_node)
        self.graph.add_node("generate_cover_letter_body", self.generate_cover_letter_body_node)
        self.graph.add_node("create_documents", self.create_documents_node)
        self.graph.add_node("finalize_workflow", self.finalize_workflow_node)
        self.graph.add_node("handle_error", self.handle_error_node)
        
        # Define the workflow edges
        self.graph.add_edge(START, "initialize_workflow")
        self.graph.add_conditional_edges(
            "initialize_workflow",
            self.should_continue_after_init,
            {
                "continue": "process_documents",
                "error": "handle_error"
            }
        )
        self.graph.add_conditional_edges(
            "process_documents", 
            self.should_continue_after_rag,
            {
                "continue": "generate_resume_summary",
                "error": "handle_error"
            }
        )
        self.graph.add_conditional_edges(
            "generate_resume_summary",
            self.should_continue_after_resume_summary,
            {
                "continue": "extract_resume_projects",
                "error": "handle_error"
            }
        )
        self.graph.add_conditional_edges(
            "extract_resume_projects",
            self.should_continue_after_resume_projects,
            {
                "continue": "generate_cover_letter_intro",
                "error": "handle_error"
            }
        )
        self.graph.add_conditional_edges(
            "generate_cover_letter_intro",
            self.should_continue_after_cover_letter_intro,
            {
                "continue": "generate_cover_letter_conclusion",
                "error": "handle_error"
            }
        )
        self.graph.add_conditional_edges(
            "generate_cover_letter_conclusion",
            self.should_continue_after_cover_letter_conclusion,
            {
                "continue": "generate_cover_letter_body",
                "error": "handle_error"
            }
        )
        self.graph.add_conditional_edges(
            "generate_cover_letter_body",
            self.should_continue_after_cover_letter_body,
            {
                "continue": "create_documents",
                "error": "handle_error"
            }
        )
        self.graph.add_conditional_edges(
            "create_documents",
            self.should_continue_after_documents,
            {
                "continue": "finalize_workflow",
                "error": "handle_error"
            }
        )
        self.graph.add_edge("finalize_workflow", END)
        self.graph.add_edge("handle_error", END)
        
        # Compile the graph
        self.compiled_graph = self.graph.compile()
        logger.info("LangGraph workflow compiled successfully")
    
    @timing_decorator
    @error_handler(context="Workflow initialization")
    def initialize_workflow_node(self, state: AppState) -> AppState:
        """
        Initialize the workflow with validation and setup.
        
        Args:
            state: Current application state
            
        Returns:
            Updated state with initialization results
        """
        logger.info("Initializing workflow...")
        
        # Update progress
        progress = state["workflow_progress"]
        progress.current_node = "initialize_workflow"
        progress.status = WorkflowStatus.RUNNING
        progress.start_time = datetime.now()
        
        # Validate required inputs
        required_fields = ["api_key", "master_resume_pdf_path", "job_description_pdf_path"]
        missing_fields = [field for field in required_fields if not state.get(field)]
        
        if missing_fields:
            error_msg = f"Missing required fields: {missing_fields}"
            logger.error(error_msg)
            return {
                **state,
                "errors": [error_msg],
                "workflow_progress": WorkflowProgress(
                    current_node="initialize_workflow",
                    completed_nodes=[],
                    failed_nodes=["initialize_workflow"],
                    total_nodes=5,
                    progress_percentage=0.0,
                    status=WorkflowStatus.FAILED,
                    error_message=error_msg
                )
            }
        
        # Validate file paths exist
        for path_key in ["master_resume_pdf_path", "job_description_pdf_path"]:
            if not os.path.exists(state[path_key]):
                error_msg = f"File not found: {state[path_key]}"
                logger.error(error_msg)
                return {
                    **state,
                    "errors": [error_msg],
                    "workflow_progress": WorkflowProgress(
                        current_node="initialize_workflow",
                        completed_nodes=[],
                        failed_nodes=["initialize_workflow"],
                        total_nodes=5,
                        progress_percentage=0.0,
                        status=WorkflowStatus.FAILED,
                        error_message=error_msg
                    )
                }
        
        # Mark initialization as complete
        progress.completed_nodes.append("initialize_workflow")
        progress.progress_percentage = 20.0
        
        logger.info("Workflow initialization completed successfully")
        return {
            **state,
            "logs": ["Workflow initialization completed"],
            "node_statuses": {"initialize_workflow": NodeStatus.COMPLETED},
            "workflow_progress": progress
        }
    
    @timing_decorator 
    @error_handler(context="Document processing")
    def process_documents_node(self, state: AppState) -> AppState:
        """
        Process documents using RAG pipeline.
        
        Args:
            state: Current application state
            
        Returns:
            Updated state with RAG processing results
        """
        logger.info("Processing documents with RAG pipeline...")
        
        progress = state["workflow_progress"]
        progress.current_node = "process_documents"
        
        try:
            # Initialize RAG processor
            rag_processor = RAGProcessor(api_key=state["api_key"])
            
            # Process master resume
            logger.info("Processing master resume...")
            master_result = rag_processor.process_pdf(
                state["master_resume_pdf_path"], 
                "master_resume"
            )
            
            # Process job description
            logger.info("Processing job description...")
            job_desc_result = rag_processor.process_pdf(
                state["job_description_pdf_path"],
                "job_description"
            )
            
            # Update progress
            progress.completed_nodes.append("process_documents")
            progress.progress_percentage = 40.0
            
            logger.info("Document processing completed successfully")
            return {
                **state,
                "rag_processor": rag_processor,
                "master_resume_result": master_result,
                "job_description_result": job_desc_result,
                "logs": ["Document processing completed"],
                "node_statuses": {"process_documents": NodeStatus.COMPLETED},
                "workflow_progress": progress
            }
            
        except Exception as e:
            error_msg = f"Document processing failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            progress.failed_nodes.append("process_documents")
            progress.status = WorkflowStatus.FAILED
            progress.error_message = error_msg
            
            return {
                **state,
                "errors": [error_msg],
                "node_statuses": {"process_documents": NodeStatus.FAILED},
                "workflow_progress": progress
            }
    
    @timing_decorator
    @error_handler(context="Content generation")
    def generate_resume_summary_node(self, state: AppState) -> AppState:
        """
        Generate resume summary using LLM.
        
        Args:
            state: Current application state
            
        Returns:
            Updated state with generated resume summary
        """
        logger.info("Generating resume summary...")
        
        progress = state["workflow_progress"]
        progress.current_node = "generate_resume_summary"
        
        try:
            # Initialize content generator
            content_generator = ContentGenerator(api_key=state["api_key"])
            
            # Get relevant context from RAG
            rag_processor = state["rag_processor"]
            
            # Generate resume summary
            logger.info("Generating resume summary...")
            resume_summary = content_generator.generate_resume_summary(
                job_description=state["job_description_result"]["text"],
                master_resume_text=state["master_resume_result"]["text"],
                rag_context=rag_processor.get_relevant_context(
                    query="technical skills, work experience, projects, education",
                    k=10
                )
            )
            
            # Update progress
            progress.completed_nodes.append("generate_resume_summary")
            progress.progress_percentage = 50.0
            
            logger.info("Resume summary generation completed successfully")
            return {
                **state,
                "content_generator": content_generator,
                "resume_summary": resume_summary,
                "logs": ["Resume summary generation completed"],
                "node_statuses": {"generate_resume_summary": NodeStatus.COMPLETED},
                "workflow_progress": progress
            }
            
        except Exception as e:
            error_msg = f"Resume summary generation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            progress.failed_nodes.append("generate_resume_summary")
            progress.status = WorkflowStatus.FAILED
            progress.error_message = error_msg
            
            return {
                **state,
                "errors": [error_msg],
                "node_statuses": {"generate_resume_summary": NodeStatus.FAILED},
                "workflow_progress": progress
            }
    
    @timing_decorator
    @error_handler(context="Content generation")
    def extract_resume_projects_node(self, state: AppState) -> AppState:
        """
        Extract resume projects using LLM.
        
        Args:
            state: Current application state
            
        Returns:
            Updated state with extracted resume projects
        """
        logger.info("Extracting resume projects...")
        
        progress = state["workflow_progress"]
        progress.current_node = "extract_resume_projects"
        
        try:
            # Initialize content generator
            content_generator = ContentGenerator(api_key=state["api_key"])
            
            # Get relevant context from RAG
            rag_processor = state["rag_processor"]
            
            # Extract resume projects
            logger.info("Extracting resume projects...")
            resume_projects = content_generator.extract_resume_projects(
                job_description=state["job_description_result"]["text"],
                master_resume_text=state["master_resume_result"]["text"],
                rag_context=rag_processor.get_relevant_context(
                    query="technical skills, work experience, projects, education",
                    k=10
                )
            )
            
            # Update progress
            progress.completed_nodes.append("extract_resume_projects")
            progress.progress_percentage = 60.0
            
            logger.info("Resume projects extraction completed successfully")
            return {
                **state,
                "content_generator": content_generator,
                "resume_projects": resume_projects,
                "logs": ["Resume projects extraction completed"],
                "node_statuses": {"extract_resume_projects": NodeStatus.COMPLETED},
                "workflow_progress": progress
            }
            
        except Exception as e:
            error_msg = f"Resume projects extraction failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            progress.failed_nodes.append("extract_resume_projects")
            progress.status = WorkflowStatus.FAILED
            progress.error_message = error_msg
            
            return {
                **state,
                "errors": [error_msg],
                "node_statuses": {"extract_resume_projects": NodeStatus.FAILED},
                "workflow_progress": progress
            }
    
    @timing_decorator
    @error_handler(context="Content generation")
    def generate_cover_letter_intro_node(self, state: AppState) -> AppState:
        """
        Generate cover letter introduction using LLM.
        
        Args:
            state: Current application state
            
        Returns:
            Updated state with generated cover letter introduction
        """
        logger.info("Generating cover letter introduction...")
        
        progress = state["workflow_progress"]
        progress.current_node = "generate_cover_letter_intro"
        
        try:
            # Initialize content generator
            content_generator = ContentGenerator(api_key=state["api_key"])
            
            # Get relevant context from RAG
            rag_processor = state["rag_processor"]
            
            # Generate cover letter introduction
            logger.info("Generating cover letter introduction...")
            cover_letter_intro = content_generator.generate_cover_letter_intro(
                job_description=state["job_description_result"]["text"],
                master_resume_text=state["master_resume_result"]["text"],
                rag_context=rag_processor.get_relevant_context(
                    query="achievements, motivation, company fit, career goals",
                    k=8
                )
            )
            
            # Update progress
            progress.completed_nodes.append("generate_cover_letter_intro")
            progress.progress_percentage = 70.0
            
            logger.info("Cover letter introduction generation completed successfully")
            return {
                **state,
                "content_generator": content_generator,
                "cover_letter_intro": cover_letter_intro,
                "logs": ["Cover letter introduction generation completed"],
                "node_statuses": {"generate_cover_letter_intro": NodeStatus.COMPLETED},
                "workflow_progress": progress
            }
            
        except Exception as e:
            error_msg = f"Cover letter introduction generation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            progress.failed_nodes.append("generate_cover_letter_intro")
            progress.status = WorkflowStatus.FAILED
            progress.error_message = error_msg
            
            return {
                **state,
                "errors": [error_msg],
                "node_statuses": {"generate_cover_letter_intro": NodeStatus.FAILED},
                "workflow_progress": progress
            }
    
    @timing_decorator
    @error_handler(context="Content generation")
    def generate_cover_letter_conclusion_node(self, state: AppState) -> AppState:
        """
        Generate cover letter conclusion using LLM.
        
        Args:
            state: Current application state
            
        Returns:
            Updated state with generated cover letter conclusion
        """
        logger.info("Generating cover letter conclusion...")
        
        progress = state["workflow_progress"]
        progress.current_node = "generate_cover_letter_conclusion"
        
        try:
            # Initialize content generator
            content_generator = ContentGenerator(api_key=state["api_key"])
            
            # Get relevant context from RAG
            rag_processor = state["rag_processor"]
            
            # Generate cover letter conclusion
            logger.info("Generating cover letter conclusion...")
            cover_letter_conclusion = content_generator.generate_cover_letter_conclusion(
                job_description=state["job_description_result"]["text"],
                master_resume_text=state["master_resume_result"]["text"],
                rag_context=rag_processor.get_relevant_context(
                    query="achievements, motivation, company fit, career goals",
                    k=8
                )
            )
            
            # Update progress
            progress.completed_nodes.append("generate_cover_letter_conclusion")
            progress.progress_percentage = 80.0
            
            logger.info("Cover letter conclusion generation completed successfully")
            return {
                **state,
                "content_generator": content_generator,
                "cover_letter_conclusion": cover_letter_conclusion,
                "logs": ["Cover letter conclusion generation completed"],
                "node_statuses": {"generate_cover_letter_conclusion": NodeStatus.COMPLETED},
                "workflow_progress": progress
            }
            
        except Exception as e:
            error_msg = f"Cover letter conclusion generation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            progress.failed_nodes.append("generate_cover_letter_conclusion")
            progress.status = WorkflowStatus.FAILED
            progress.error_message = error_msg
            
            return {
                **state,
                "errors": [error_msg],
                "node_statuses": {"generate_cover_letter_conclusion": NodeStatus.FAILED},
                "workflow_progress": progress
            }
    
    @timing_decorator
    @error_handler(context="Content generation")
    def generate_cover_letter_body_node(self, state: AppState) -> AppState:
        """
        Generate cover letter body using LLM.
        
        Args:
            state: Current application state
            
        Returns:
            Updated state with generated cover letter body
        """
        logger.info("Generating cover letter body...")
        
        progress = state["workflow_progress"]
        progress.current_node = "generate_cover_letter_body"
        
        try:
            # Initialize content generator
            content_generator = ContentGenerator(api_key=state["api_key"])
            
            # Get relevant context from RAG
            rag_processor = state["rag_processor"]
            
            # Generate cover letter body
            logger.info("Generating cover letter body...")
            cover_letter_body = content_generator.generate_cover_letter_body(
                job_description=state["job_description_result"]["text"],
                master_resume_text=state["master_resume_result"]["text"],
                rag_context=rag_processor.get_relevant_context(
                    query="achievements, motivation, company fit, career goals",
                    k=8
                )
            )
            
            # Update progress
            progress.completed_nodes.append("generate_cover_letter_body")
            progress.progress_percentage = 90.0
            
            logger.info("Cover letter body generation completed successfully")
            return {
                **state,
                "content_generator": content_generator,
                "cover_letter_body": cover_letter_body,
                "logs": ["Cover letter body generation completed"],
                "node_statuses": {"generate_cover_letter_body": NodeStatus.COMPLETED},
                "workflow_progress": progress
            }
            
        except Exception as e:
            error_msg = f"Cover letter body generation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            progress.failed_nodes.append("generate_cover_letter_body")
            progress.status = WorkflowStatus.FAILED
            progress.error_message = error_msg
            
            return {
                **state,
                "errors": [error_msg],
                "node_statuses": {"generate_cover_letter_body": NodeStatus.FAILED},
                "workflow_progress": progress
            }
    
    @timing_decorator
    @error_handler(context="Document creation")
    def create_documents_node(self, state: AppState) -> AppState:
        """
        Create final DOCX and PDF documents.
        
        Args:
            state: Current application state
            
        Returns:
            Updated state with generated documents
        """
        logger.info("Creating final documents...")
        
        progress = state["workflow_progress"]
        progress.current_node = "create_documents"
        
        try:
            # Initialize document generator
            doc_generator = DocumentGenerator()
            
            # Generate filename prefix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = f"tailored_{timestamp}"
            
            # Parse resume content properly
            resume_summary_content = ""
            if state.get("resume_summary") and hasattr(state["resume_summary"], "content"):
                resume_summary_content = state["resume_summary"].content
            
            resume_projects_content = ""
            if state.get("resume_projects") and hasattr(state["resume_projects"], "content"):
                resume_projects_content = state["resume_projects"].content
            
            resume_content_dict = {
                "summary": resume_summary_content,
                "project_title": "Key Projects",
                "project_bullets": resume_projects_content
            }
            
            # Validate resume content
            resume_validation = validate_resume(resume_content_dict)
            if not resume_validation.is_valid:
                logger.warning(f"Resume validation issues: {resume_validation.issues}")
                for suggestion in resume_validation.suggestions:
                    logger.info(f"Resume suggestion: {suggestion}")
            
            logger.info(f"Resume metrics: {resume_validation.metrics.word_count} words, "
                       f"{resume_validation.metrics.estimated_pages:.1f} pages")
            
            # Parse cover letter content properly
            cover_letter_intro_content = ""
            if state.get("cover_letter_intro") and hasattr(state["cover_letter_intro"], "content"):
                cover_letter_intro_content = state["cover_letter_intro"].content
            
            cover_letter_body_content = ""
            if state.get("cover_letter_body") and hasattr(state["cover_letter_body"], "content"):
                cover_letter_body_content = state["cover_letter_body"].content
                
            cover_letter_conclusion_content = ""
            if state.get("cover_letter_conclusion") and hasattr(state["cover_letter_conclusion"], "content"):
                cover_letter_conclusion_content = state["cover_letter_conclusion"].content
            
            cover_letter_content_dict = {
                "introduction": cover_letter_intro_content,
                "body": cover_letter_body_content,
                "conclusion": cover_letter_conclusion_content
            }
            
            # Validate cover letter content
            cover_letter_validation = validate_cover_letter(cover_letter_content_dict)
            if not cover_letter_validation.is_valid:
                logger.warning(f"Cover letter validation issues: {cover_letter_validation.issues}")
                for suggestion in cover_letter_validation.suggestions:
                    logger.info(f"Cover letter suggestion: {suggestion}")
            
            logger.info(f"Cover letter metrics: {cover_letter_validation.metrics.word_count} words, "
                       f"{cover_letter_validation.metrics.estimated_pages:.1f} pages")
            
            # Create resume document
            logger.info("Creating resume document...")
            resume_doc = doc_generator.generate_resume(
                content=resume_content_dict,
                output_filename=f"{prefix}_resume"
            )
            
            # Create cover letter document
            logger.info("Creating cover letter document...")
            cover_letter_doc = doc_generator.generate_cover_letter(
                content=cover_letter_content_dict,
                output_filename=f"{prefix}_cover_letter"
            )
            
            # Update progress
            progress.completed_nodes.append("create_documents")
            progress.progress_percentage = 95.0
            
            logger.info("Document creation completed successfully")
            return {
                **state,
                "document_generator": doc_generator,
                "resume_document": resume_doc,
                "cover_letter_document": cover_letter_doc,
                "logs": ["Document creation completed"],
                "node_statuses": {"create_documents": NodeStatus.COMPLETED},
                "workflow_progress": progress
            }
            
        except Exception as e:
            error_msg = f"Document creation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            progress.failed_nodes.append("create_documents")
            progress.status = WorkflowStatus.FAILED
            progress.error_message = error_msg
            
            return {
                **state,
                "errors": [error_msg],
                "node_statuses": {"create_documents": NodeStatus.FAILED},
                "workflow_progress": progress
            }
    
    @timing_decorator
    @error_handler(context="Workflow finalization")
    def finalize_workflow_node(self, state: AppState) -> AppState:
        """
        Finalize the workflow and prepare results.
        
        Args:
            state: Current application state
            
        Returns:
            Final state with workflow completion
        """
        logger.info("Finalizing workflow...")
        
        progress = state["workflow_progress"]
        progress.current_node = "finalize_workflow"
        progress.completed_nodes.append("finalize_workflow")
        progress.progress_percentage = 100.0
        progress.status = WorkflowStatus.COMPLETED
        progress.end_time = datetime.now()
        
        # Log final statistics
        total_time = progress.end_time - progress.start_time
        logger.info(f"Workflow completed successfully in {total_time.total_seconds():.2f} seconds")
        
        return {
            **state,
            "logs": [f"Workflow completed successfully in {total_time.total_seconds():.2f} seconds"],
            "node_statuses": {"finalize_workflow": NodeStatus.COMPLETED},
            "workflow_progress": progress
        }
    
    @error_handler(context="Error handling")
    def handle_error_node(self, state: AppState) -> AppState:
        """
        Handle workflow errors and cleanup.
        
        Args:
            state: Current application state
            
        Returns:
            State with error handling completed
        """
        logger.error("Handling workflow error...")
        
        progress = state["workflow_progress"]
        progress.status = WorkflowStatus.FAILED
        progress.end_time = datetime.now()
        
        # Log error details
        if state.get("errors"):
            for error in state["errors"]:
                logger.error(f"Workflow error: {error}")
        
        return {
            **state,
            "logs": ["Workflow failed - see errors for details"],
            "workflow_progress": progress
        }
    
    # Conditional edge functions
    def should_continue_after_init(self, state: AppState) -> str:
        """Determine next step after initialization."""
        if state["workflow_progress"].status == WorkflowStatus.FAILED:
            return "error"
        return "continue"
    
    def should_continue_after_rag(self, state: AppState) -> str:
        """Determine next step after RAG processing.""" 
        if state["workflow_progress"].status == WorkflowStatus.FAILED:
            return "error"
        return "continue"
    
    def should_continue_after_resume_summary(self, state: AppState) -> str:
        """Determine next step after resume summary generation."""
        if state["workflow_progress"].status == WorkflowStatus.FAILED:
            return "error"
        return "continue"
    
    def should_continue_after_resume_projects(self, state: AppState) -> str:
        """Determine next step after resume projects extraction."""
        if state["workflow_progress"].status == WorkflowStatus.FAILED:
            return "error"
        return "continue"
    
    def should_continue_after_cover_letter_intro(self, state: AppState) -> str:
        """Determine next step after cover letter intro generation."""
        if state["workflow_progress"].status == WorkflowStatus.FAILED:
            return "error"
        return "continue"
    
    def should_continue_after_cover_letter_conclusion(self, state: AppState) -> str:
        """Determine next step after cover letter conclusion generation."""
        if state["workflow_progress"].status == WorkflowStatus.FAILED:
            return "error"
        return "continue"
    
    def should_continue_after_cover_letter_body(self, state: AppState) -> str:
        """Determine next step after cover letter body generation."""
        if state["workflow_progress"].status == WorkflowStatus.FAILED:
            return "error"
        return "continue"
    
    def should_continue_after_documents(self, state: AppState) -> str:
        """Determine next step after document creation."""
        if state["workflow_progress"].status == WorkflowStatus.FAILED:
            return "error"
        return "continue"
    
    @timing_decorator
    def execute_workflow(self, 
                        api_key: str,
                        master_resume_path: str, 
                        job_description_path: str,
                        user_preferences: Optional[Dict[str, Any]] = None,
                        session_id: Optional[str] = None) -> AppState:
        """
        Execute the complete workflow.
        
        Args:
            api_key: Google API key for Gemini
            master_resume_path: Path to master resume PDF
            job_description_path: Path to job description PDF
            user_preferences: Optional user preferences
            session_id: Optional session ID for tracking
            
        Returns:
            Final workflow state
        """
        logger.info("Starting Application Factory workflow execution...")
        
        # Create initial state
        initial_state: AppState = {
            "api_key": api_key,
            "master_resume_pdf_path": master_resume_path,
            "job_description_pdf_path": job_description_path,
            "rag_processor": None,
            "master_resume_result": None,
            "job_description_result": None,
            "content_generator": None,
            "resume_content": None,
            "cover_letter_content": None,
            "document_generator": None,
            "resume_document": None,
            "cover_letter_document": None,
            "workflow_progress": WorkflowProgress(
                current_node="",
                completed_nodes=[],
                failed_nodes=[],
                total_nodes=5,
                progress_percentage=0.0,
                status=WorkflowStatus.PENDING
            ),
            "node_statuses": {},
            "errors": [],
            "logs": [],
            "user_preferences": user_preferences or {},
            "session_id": session_id or SessionManager.get_timestamp(),
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Execute the workflow
            final_state = self.compiled_graph.invoke(initial_state)
            
            # Log execution summary
            if final_state["workflow_progress"].status == WorkflowStatus.COMPLETED:
                logger.info("Workflow executed successfully")
            else:
                logger.error("Workflow execution failed")
                
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution error: {str(e)}", exc_info=True)
            
            # Return error state
            error_state = initial_state.copy()
            error_state["errors"] = [f"Workflow execution error: {str(e)}"]
            error_state["workflow_progress"].status = WorkflowStatus.FAILED
            error_state["workflow_progress"].error_message = str(e)
            error_state["workflow_progress"].end_time = datetime.now()
            
            return error_state
    
    def get_workflow_status(self, state: AppState) -> Dict[str, Any]:
        """
        Get current workflow status and progress.
        
        Args:
            state: Current application state
            
        Returns:
            Status information dictionary
        """
        progress = state["workflow_progress"]
        return {
            "status": progress.status.value,
            "current_node": progress.current_node,
            "progress_percentage": progress.progress_percentage,
            "completed_nodes": progress.completed_nodes,
            "failed_nodes": progress.failed_nodes,
            "errors": state.get("errors", []),
            "logs": state.get("logs", []),
            "start_time": progress.start_time.isoformat() if progress.start_time else None,
            "end_time": progress.end_time.isoformat() if progress.end_time else None,
            "total_nodes": progress.total_nodes
        }


# Convenience functions for external use
def create_workflow() -> ApplicationFactoryWorkflow:
    """Create a new workflow instance."""
    return ApplicationFactoryWorkflow()


def execute_application_factory(api_key: str,
                              master_resume_path: str,
                              job_description_path: str, 
                              user_preferences: Optional[Dict[str, Any]] = None,
                              session_id: Optional[str] = None) -> AppState:
    """
    Convenience function to execute the complete Application Factory workflow.
    
    Args:
        api_key: Google API key for Gemini
        master_resume_path: Path to master resume PDF
        job_description_path: Path to job description PDF
        user_preferences: Optional user preferences
        session_id: Optional session ID for tracking
        
    Returns:
        Final workflow state
    """
    workflow = create_workflow()
    return workflow.execute_workflow(
        api_key=api_key,
        master_resume_path=master_resume_path,
        job_description_path=job_description_path,
        user_preferences=user_preferences,
        session_id=session_id
    ) 