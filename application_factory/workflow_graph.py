"""
LangGraph Workflow Implementation for the Application Factory - Phase 5.

This module implements a sophisticated workflow using LangGraph to orchestrate
the entire application generation process with rate limiting and optimized
LLM interactions for maximum accuracy and efficiency.
"""

import logging
import time
from typing import Dict, List, Optional, Any, TypedDict
from datetime import datetime, timedelta
from pathlib import Path
import json

from langgraph.graph import StateGraph, END

from state_rag import (
    GraphStateRAG, create_initial_state, update_state_stage,
    set_error_state, clear_error_state, validate_state,
    update_vector_store_path, update_retrieved_contexts,
    update_ai_generated_resume_sections, update_ai_generated_cover_letter,
    update_extracted_job_skills, update_llm_initialization_status,
    update_pdf_paths, get_llm_status, validate_state_for_llm_generation
)
from rag_utils import RAGManager, create_resume_vector_store
from llm_utils import LLMManager, create_llm_manager
from html_pdf_utils import generate_resume_pdf, generate_cover_letter_pdf
from config.paths import PathManager

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Advanced rate limiter for LLM API calls.
    Implements sliding window rate limiting with 15 requests per minute.
    """
    
    def __init__(self, max_requests: int = 15, time_window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed per time window
            time_window: Time window in seconds (default: 60 for per minute)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def _clean_old_requests(self):
        """Remove requests older than the time window."""
        current_time = time.time()
        self.requests = [req_time for req_time in self.requests 
                        if current_time - req_time < self.time_window]
    
    def can_make_request(self) -> bool:
        """Check if a request can be made without hitting the rate limit."""
        self._clean_old_requests()
        return len(self.requests) < self.max_requests
    
    def wait_time_until_available(self) -> float:
        """Get the time to wait until a request can be made."""
        if self.can_make_request():
            return 0.0
        
        self._clean_old_requests()
        if len(self.requests) == 0:
            return 0.0
        
        # Calculate when the oldest request will expire
        oldest_request = min(self.requests)
        wait_time = self.time_window - (time.time() - oldest_request)
        return max(0.0, wait_time)
    
    def record_request(self):
        """Record a request being made."""
        self.requests.append(time.time())
        logger.debug(f"Request recorded. Current count: {len(self.requests)}/{self.max_requests}")
    
    def wait_if_needed(self):
        """Wait if necessary before making a request."""
        wait_time = self.wait_time_until_available()
        if wait_time > 0:
            logger.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        
        self.record_request()


class WorkflowMetrics:
    """Track workflow execution metrics and performance."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.node_times = {}
        self.api_calls = 0
        self.errors = []
        self.completed_nodes = []
    
    def start_workflow(self):
        """Start timing the workflow."""
        self.start_time = time.time()
        logger.info("Workflow execution started")
    
    def end_workflow(self):
        """End timing the workflow."""
        self.end_time = time.time()
        total_time = self.end_time - self.start_time if self.start_time else 0
        logger.info(f"Workflow completed in {total_time:.2f} seconds")
    
    def start_node(self, node_name: str):
        """Start timing a node execution."""
        self.node_times[node_name] = {"start": time.time()}
    
    def end_node(self, node_name: str):
        """End timing a node execution."""
        if node_name in self.node_times:
            self.node_times[node_name]["end"] = time.time()
            duration = self.node_times[node_name]["end"] - self.node_times[node_name]["start"]
            self.node_times[node_name]["duration"] = duration
            self.completed_nodes.append(node_name)
            logger.info(f"Node '{node_name}' completed in {duration:.2f} seconds")
    
    def record_api_call(self):
        """Record an API call."""
        self.api_calls += 1
    
    def record_error(self, node_name: str, error: str):
        """Record an error."""
        self.errors.append({"node": node_name, "error": error, "timestamp": time.time()})
        logger.error(f"Error in node '{node_name}': {error}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get workflow execution summary."""
        total_time = (self.end_time - self.start_time) if (self.start_time and self.end_time) else 0
        
        return {
            "total_time": total_time,
            "completed_nodes": len(self.completed_nodes),
            "api_calls": self.api_calls,
            "errors": len(self.errors),
            "node_performance": {
                name: data.get("duration", 0) for name, data in self.node_times.items()
            },
            "success_rate": (len(self.completed_nodes) - len(self.errors)) / max(1, len(self.completed_nodes))
        }


class ApplicationFactoryWorkflow:
    """
    Main LangGraph workflow for the Application Factory.
    Orchestrates the entire process from resume analysis to PDF generation.
    """
    
    def __init__(self, checkpoint_dir: str = "temp/checkpoints"):
        """
        Initialize the workflow.
        
        Args:
            checkpoint_dir: Directory for storing workflow checkpoints
        """
        self.rate_limiter = RateLimiter(max_requests=15, time_window=60)
        self.metrics = WorkflowMetrics()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize managers
        self.rag_manager = None
        self.llm_manager = None
        self.path_manager = None
        
        # Build the workflow graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(GraphStateRAG)
        
        # Add nodes
        workflow.add_node("initialize", self.initialize_node)
        workflow.add_node("create_vector_store", self.create_vector_store_node)
        workflow.add_node("extract_job_skills", self.extract_job_skills_node)
        workflow.add_node("generate_resume_sections", self.generate_resume_sections_node)
        workflow.add_node("generate_cover_letter", self.generate_cover_letter_node)
        workflow.add_node("enhance_content", self.enhance_content_node)
        workflow.add_node("generate_pdfs", self.generate_pdfs_node)
        workflow.add_node("finalize", self.finalize_node)
        
        # Add error handling node
        workflow.add_node("handle_error", self.handle_error_node)
        
        # Define the workflow edges
        workflow.set_entry_point("initialize")
        
        # Main workflow path
        workflow.add_edge("initialize", "create_vector_store")
        workflow.add_edge("create_vector_store", "extract_job_skills")
        workflow.add_edge("extract_job_skills", "generate_resume_sections")
        workflow.add_edge("generate_resume_sections", "generate_cover_letter")
        workflow.add_edge("generate_cover_letter", "enhance_content")
        workflow.add_edge("enhance_content", "generate_pdfs")
        workflow.add_edge("generate_pdfs", "finalize")
        workflow.add_edge("finalize", END)
        
        # Error handling edges
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def initialize_node(self, state: GraphStateRAG) -> GraphStateRAG:
        """Initialize the workflow and required managers."""
        node_name = "initialize"
        self.metrics.start_node(node_name)
        
        try:
            logger.info("Initializing Application Factory Workflow")
            
            # Update stage
            updated_state = update_state_stage(state, "initialization")
            
            # Initialize managers
            self.rag_manager = RAGManager()
            self.llm_manager = LLMManager(self.rag_manager)
            self.path_manager = PathManager(updated_state.get("job_specific_output_folder_path", "generated_applications"))
            
            # Update LLM initialization status
            updated_state = update_llm_initialization_status(updated_state, True)
            
            # Validate input data
            validation = validate_state_for_llm_generation(updated_state)
            if not validation["ready"]:
                raise ValueError(f"State validation failed: {validation['issues']}")
            
            updated_state = clear_error_state(updated_state)
            logger.info("✅ Workflow initialization completed successfully")
            
            self.metrics.end_node(node_name)
            return updated_state
            
        except Exception as e:
            self.metrics.record_error(node_name, str(e))
            logger.error(f"Failed to initialize workflow: {e}")
            return set_error_state(state, f"Initialization failed: {e}", node_name)
    
    def create_vector_store_node(self, state: GraphStateRAG) -> GraphStateRAG:
        """Create vector store from master resume."""
        node_name = "create_vector_store"
        self.metrics.start_node(node_name)
        
        try:
            if state.get("error_message"):
                return state
            
            logger.info("Creating vector store from master resume")
            updated_state = update_state_stage(state, "vector_store_creation")
            
            # Create vector store
            master_resume_content = updated_state.get("master_resume_content", "")
            if not master_resume_content:
                raise ValueError("Master resume content is required")
            
            # Create vector store path
            vector_store_path = create_resume_vector_store(
                master_resume_content,
                save_path=self.path_manager.get_vector_store_path()
            )
            
            if vector_store_path:
                updated_state = update_vector_store_path(updated_state, str(vector_store_path))
                logger.info(f"✅ Vector store created: {vector_store_path}")
            else:
                logger.warning("⚠️ Vector store creation failed, continuing without RAG enhancement")
            
            self.metrics.end_node(node_name)
            return updated_state
            
        except Exception as e:
            self.metrics.record_error(node_name, str(e))
            logger.error(f"Vector store creation failed: {e}")
            return set_error_state(state, f"Vector store creation failed: {e}", node_name)
    
    def extract_job_skills_node(self, state: GraphStateRAG) -> GraphStateRAG:
        """Extract and categorize skills from job description."""
        node_name = "extract_job_skills"
        self.metrics.start_node(node_name)
        
        try:
            if state.get("error_message"):
                return state
            
            logger.info("Extracting job skills and requirements")
            updated_state = update_state_stage(state, "job_analysis")
            
            # Rate limiting
            self.rate_limiter.wait_if_needed()
            self.metrics.record_api_call()
            
            # Extract job skills
            job_description = updated_state.get("job_description_content", "")
            skills_data = self.llm_manager.extract_job_skills(job_description)
            
            # Update state with extracted skills
            updated_state = update_extracted_job_skills(updated_state, skills_data)
            
            total_skills = sum(len(skills) for skills in skills_data.values())
            logger.info(f"✅ Extracted {total_skills} skills in {len(skills_data)} categories")
            
            self.metrics.end_node(node_name)
            return updated_state
            
        except Exception as e:
            self.metrics.record_error(node_name, str(e))
            logger.error(f"Job skills extraction failed: {e}")
            return set_error_state(state, f"Job skills extraction failed: {e}", node_name)
    
    def generate_resume_sections_node(self, state: GraphStateRAG) -> GraphStateRAG:
        """Generate all resume sections using LLM."""
        node_name = "generate_resume_sections"
        self.metrics.start_node(node_name)
        
        try:
            if state.get("error_message"):
                return state
            
            logger.info("Generating AI-powered resume sections")
            updated_state = update_state_stage(state, "resume_generation")
            
            # Get input data
            job_description = updated_state.get("job_description_content", "")
            master_resume = updated_state.get("master_resume_content", "")
            
            # Generate each section with rate limiting
            sections = ['summary', 'skills', 'education', 'experience', 'projects']
            generated_sections = {}
            
            for section in sections:
                try:
                    # Rate limiting
                    self.rate_limiter.wait_if_needed()
                    self.metrics.record_api_call()
                    
                    # Generate section
                    content = self.llm_manager.generate_resume_section(
                        section, job_description, master_resume
                    )
                    
                    if content and len(content.strip()) > 20:
                        generated_sections[section] = content
                        logger.info(f"✅ Generated {section} section ({len(content)} chars)")
                    else:
                        logger.warning(f"⚠️ Generated {section} section seems too short")
                        generated_sections[section] = f"Generated {section} content placeholder"
                
                except Exception as e:
                    logger.error(f"Failed to generate {section} section: {e}")
                    generated_sections[section] = f"Error generating {section} section"
            
            # Update state
            updated_state = update_ai_generated_resume_sections(updated_state, generated_sections)
            
            logger.info(f"✅ Generated {len(generated_sections)} resume sections")
            
            self.metrics.end_node(node_name)
            return updated_state
            
        except Exception as e:
            self.metrics.record_error(node_name, str(e))
            logger.error(f"Resume section generation failed: {e}")
            return set_error_state(state, f"Resume section generation failed: {e}", node_name)
    
    def generate_cover_letter_node(self, state: GraphStateRAG) -> GraphStateRAG:
        """Generate cover letter using LLM."""
        node_name = "generate_cover_letter"
        self.metrics.start_node(node_name)
        
        try:
            if state.get("error_message"):
                return state
            
            logger.info("Generating AI-powered cover letter")
            updated_state = update_state_stage(state, "cover_letter_generation")
            
            # Rate limiting
            self.rate_limiter.wait_if_needed()
            self.metrics.record_api_call()
            
            # Extract company and position from job description
            job_description = updated_state.get("job_description_content", "")
            company, position = self.path_manager.extract_company_and_position(job_description)
            
            # Generate cover letter
            applicant_name = updated_state.get("resume_header_text", "").split('\n')[0] if updated_state.get("resume_header_text") else "Applicant"
            master_resume = updated_state.get("master_resume_content", "")
            
            cover_letter_data = self.llm_manager.generate_cover_letter(
                job_description=job_description,
                company=company,
                position=position,
                applicant_name=applicant_name,
                master_resume_content=master_resume
            )
            
            # Update state
            updated_state = update_ai_generated_cover_letter(updated_state, cover_letter_data)
            
            logger.info(f"✅ Generated cover letter for {position} at {company}")
            
            self.metrics.end_node(node_name)
            return updated_state
            
        except Exception as e:
            self.metrics.record_error(node_name, str(e))
            logger.error(f"Cover letter generation failed: {e}")
            return set_error_state(state, f"Cover letter generation failed: {e}", node_name)
    
    def enhance_content_node(self, state: GraphStateRAG) -> GraphStateRAG:
        """Enhance generated content for better impact."""
        node_name = "enhance_content"
        self.metrics.start_node(node_name)
        
        try:
            if state.get("error_message"):
                return state
            
            logger.info("Enhancing content for maximum impact")
            updated_state = update_state_stage(state, "content_enhancement")
            
            # Get generated content
            resume_sections = updated_state.get("ai_generated_resume_sections", {})
            job_description = updated_state.get("job_description_content", "")
            
            # Enhance critical sections (summary and skills)
            enhanced_sections = resume_sections.copy()
            critical_sections = ['summary', 'skills']
            
            for section in critical_sections:
                if section in resume_sections:
                    try:
                        # Rate limiting
                        self.rate_limiter.wait_if_needed()
                        self.metrics.record_api_call()
                        
                        # Enhance content
                        enhanced_content = self.llm_manager.enhance_content(
                            content=resume_sections[section],
                            content_type=f"resume_{section}",
                            job_description=job_description
                        )
                        
                        if enhanced_content and len(enhanced_content) > len(resume_sections[section]) * 0.8:
                            enhanced_sections[section] = enhanced_content
                            logger.info(f"✅ Enhanced {section} section")
                        else:
                            logger.warning(f"⚠️ Enhancement for {section} didn't improve content")
                    
                    except Exception as e:
                        logger.error(f"Failed to enhance {section}: {e}")
            
            # Update state with enhanced sections
            updated_state = update_ai_generated_resume_sections(updated_state, enhanced_sections)
            
            logger.info("✅ Content enhancement completed")
            
            self.metrics.end_node(node_name)
            return updated_state
            
        except Exception as e:
            self.metrics.record_error(node_name, str(e))
            logger.error(f"Content enhancement failed: {e}")
            # Continue without enhancement rather than failing
            updated_state = update_state_stage(state, "content_enhancement_skipped")
            return updated_state
    
    def generate_pdfs_node(self, state: GraphStateRAG) -> GraphStateRAG:
        """Generate PDF files from the generated content."""
        node_name = "generate_pdfs"
        self.metrics.start_node(node_name)
        
        try:
            if state.get("error_message"):
                return state
            
            logger.info("Generating PDF files")
            updated_state = update_state_stage(state, "pdf_generation")
            
            # Get generated content
            resume_sections = updated_state.get("ai_generated_resume_sections", {})
            cover_letter_data = updated_state.get("ai_generated_cover_letter", {})
            
            # Create job folder
            job_description = updated_state.get("job_description_content", "")
            job_folder = self.path_manager.create_job_folder(job_description)
            
            # Get user name and contact info
            user_name = updated_state.get("resume_header_text", "User").split('\n')[0]
            
            # Generate resume PDF with actual contact info
            resume_path = self.path_manager.get_resume_path(job_folder, user_name)
            contact_info = updated_state.get("contact_info", {})
            
            resume_success = generate_resume_pdf(
                sections=resume_sections,
                name=contact_info.get('name', user_name),
                phone=contact_info.get('phone', "(555) 123-4567"),
                email=contact_info.get('email', f"{user_name.lower().replace(' ', '.')}@email.com"),
                linkedin=contact_info.get('linkedin', f"linkedin.com/in/{user_name.lower().replace(' ', '-')}"),
                website=contact_info.get('website', ""),
                location="Toronto, ON",
                output_path=str(resume_path)
            )
            
            # Generate cover letter PDF
            cover_letter_path = self.path_manager.get_cover_letter_path(job_folder, user_name)
            
            # Get contact info from state if available
            contact_info = updated_state.get("contact_info", {})
            header_text = f"{contact_info.get('name', user_name)}"
            if contact_info.get('phone'):
                header_text += f"\n{contact_info['phone']}"
            if contact_info.get('email'):
                header_text += f" | {contact_info['email']}"
            if contact_info.get('linkedin'):
                header_text += f" | LinkedIn"
            
            cover_letter_success = generate_cover_letter_pdf(
                intro=cover_letter_data.get("introduction", ""),
                body=cover_letter_data.get("body_paragraphs", []),
                conclusion=cover_letter_data.get("conclusion", ""),
                header_text=header_text,
                output_path=str(cover_letter_path)
            )
            
            # Update state with PDF paths
            if resume_success or cover_letter_success:
                updated_state = update_pdf_paths(
                    updated_state,
                    resume_path=str(resume_path) if resume_success else "",
                    cover_letter_path=str(cover_letter_path) if cover_letter_success else ""
                )
                
                logger.info(f"✅ PDFs generated - Resume: {resume_success}, Cover Letter: {cover_letter_success}")
            else:
                raise ValueError("Failed to generate any PDF files")
            
            self.metrics.end_node(node_name)
            return updated_state
            
        except Exception as e:
            self.metrics.record_error(node_name, str(e))
            logger.error(f"PDF generation failed: {e}")
            return set_error_state(state, f"PDF generation failed: {e}", node_name)
    
    def finalize_node(self, state: GraphStateRAG) -> GraphStateRAG:
        """Finalize the workflow and generate summary."""
        node_name = "finalize"
        self.metrics.start_node(node_name)
        
        try:
            logger.info("Finalizing workflow")
            updated_state = update_state_stage(state, "completed")
            
            # Generate workflow summary
            self.metrics.end_workflow()
            summary = self.metrics.get_summary()
            
            # Update processing metadata
            updated_state["processing_metadata"]["workflow_completed"] = True
            updated_state["processing_metadata"]["completion_timestamp"] = str(time.time())
            updated_state["processing_metadata"]["performance_summary"] = summary
            
            logger.info("🎉 Application Factory Workflow completed successfully!")
            logger.info(f"📊 Summary: {summary['total_time']:.1f}s total, {summary['api_calls']} API calls, {summary['success_rate']:.1%} success rate")
            
            self.metrics.end_node(node_name)
            return updated_state
            
        except Exception as e:
            self.metrics.record_error(node_name, str(e))
            logger.error(f"Workflow finalization failed: {e}")
            return set_error_state(state, f"Finalization failed: {e}", node_name)
    
    def handle_error_node(self, state: GraphStateRAG) -> GraphStateRAG:
        """Handle errors and generate fallback content if possible."""
        node_name = "handle_error"
        
        try:
            error_message = state.get("error_message", "Unknown error")
            logger.error(f"Handling workflow error: {error_message}")
            
            # Try to provide some fallback functionality
            # This could include generating basic templates or partial content
            
            updated_state = update_state_stage(state, "error_handled")
            
            # Record final metrics
            self.metrics.end_workflow()
            summary = self.metrics.get_summary()
            updated_state["processing_metadata"]["error_summary"] = summary
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Error handler failed: {e}")
            return state
    
    def run_workflow(self, initial_state: GraphStateRAG) -> GraphStateRAG:
        """
        Run the complete workflow.
        
        Args:
            initial_state: Initial state with input data
            
        Returns:
            Final state with all generated content
        """
        try:
            logger.info("🚀 Starting Application Factory Workflow")
            self.metrics.start_workflow()
            
            # Run the workflow
            final_state = self.graph.invoke(initial_state)
            
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return set_error_state(initial_state, f"Workflow execution failed: {e}", "workflow")


# Convenience functions for easier access
def create_workflow(checkpoint_dir: str = "temp/checkpoints") -> ApplicationFactoryWorkflow:
    """
    Create and configure an Application Factory workflow.
    
    Args:
        checkpoint_dir: Directory for storing workflow checkpoints
        
    Returns:
        Configured workflow instance
    """
    return ApplicationFactoryWorkflow(checkpoint_dir)


def run_application_generation(
    master_resume_content: str,
    job_description_content: str,
    output_base_path: str = "generated_applications",
    resume_header: str = "",
    cover_letter_header: str = ""
) -> GraphStateRAG:
    """
    Convenience function to run the complete application generation workflow.
    
    Args:
        master_resume_content: Master resume content
        job_description_content: Target job description
        output_base_path: Base output directory
        resume_header: Resume header text
        cover_letter_header: Cover letter header text
        
    Returns:
        Final workflow state with generated content
    """
    # Create initial state
    initial_state = create_initial_state(
        master_resume_path="",  # Content is provided directly
        job_description_content=job_description_content,
        output_base_path=output_base_path,
        resume_header=resume_header,
        cover_letter_header=cover_letter_header
    )
    
    # Add master resume content
    initial_state["master_resume_content"] = master_resume_content
    
    # Create and run workflow
    workflow = create_workflow()
    final_state = workflow.run_workflow(initial_state)
    
    return final_state
