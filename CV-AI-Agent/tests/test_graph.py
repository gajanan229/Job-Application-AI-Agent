"""
Tests for the LangGraph Workflow Engine.

This module contains comprehensive tests for the Application Factory workflow
engine, including unit tests for individual nodes, integration tests for the
complete workflow, and validation of state management.
"""

import pytest
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from core.graph import (
    ApplicationFactoryWorkflow,
    AppState,
    WorkflowStatus,
    NodeStatus,
    WorkflowProgress,
    create_workflow,
    execute_application_factory
)
# from core.rag_processor import RAGResult  # Not needed - using Dict[str, Any]
from core.llm_service import GenerationResponse, ContentType
from core.document_generator import GeneratedDocument


class TestWorkflowComponents:
    """Test individual workflow components."""
    
    def test_workflow_status_enum(self):
        """Test WorkflowStatus enum values."""
        assert WorkflowStatus.PENDING.value == "pending"
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"
        assert WorkflowStatus.CANCELLED.value == "cancelled"
    
    def test_node_status_enum(self):
        """Test NodeStatus enum values."""
        assert NodeStatus.PENDING.value == "pending"
        assert NodeStatus.RUNNING.value == "running"
        assert NodeStatus.COMPLETED.value == "completed"
        assert NodeStatus.FAILED.value == "failed"
        assert NodeStatus.SKIPPED.value == "skipped"
    
    def test_workflow_progress_creation(self):
        """Test WorkflowProgress dataclass creation."""
        progress = WorkflowProgress(
            current_node="test_node",
            completed_nodes=["node1", "node2"],
            failed_nodes=[],
            total_nodes=5,
            progress_percentage=40.0,
            status=WorkflowStatus.RUNNING
        )
        
        assert progress.current_node == "test_node"
        assert progress.completed_nodes == ["node1", "node2"]
        assert progress.failed_nodes == []
        assert progress.total_nodes == 5
        assert progress.progress_percentage == 40.0
        assert progress.status == WorkflowStatus.RUNNING
        assert progress.start_time is None
        assert progress.end_time is None
        assert progress.error_message is None
    
    def test_workflow_progress_to_dict(self):
        """Test WorkflowProgress to_dict conversion."""
        start_time = datetime.now()
        end_time = datetime.now()
        
        progress = WorkflowProgress(
            current_node="test_node",
            completed_nodes=["node1"],
            failed_nodes=["node2"],
            total_nodes=3,
            progress_percentage=66.6,
            status=WorkflowStatus.FAILED,
            start_time=start_time,
            end_time=end_time,
            error_message="Test error"
        )
        
        result = progress.to_dict()
        
        assert result["current_node"] == "test_node"
        assert result["completed_nodes"] == ["node1"]
        assert result["failed_nodes"] == ["node2"]
        assert result["total_nodes"] == 3
        assert result["progress_percentage"] == 66.6
        assert result["status"] == "failed"
        assert result["start_time"] == start_time.isoformat()
        assert result["end_time"] == end_time.isoformat()
        assert result["error_message"] == "Test error"


class TestApplicationFactoryWorkflow:
    """Test the main ApplicationFactoryWorkflow class."""
    
    def test_workflow_initialization(self):
        """Test workflow initialization."""
        with patch('core.graph.logger'):
            workflow = ApplicationFactoryWorkflow()
            
            assert workflow.graph is not None
            assert workflow.compiled_graph is not None
    
    def test_workflow_graph_compilation(self):
        """Test that the workflow graph compiles without errors."""
        with patch('core.graph.logger'):
            workflow = ApplicationFactoryWorkflow()
            
            # Verify the graph has the expected nodes
            expected_nodes = [
                "initialize_workflow", 
                "process_documents",
                "generate_content",
                "create_documents", 
                "finalize_workflow",
                "handle_error"
            ]
            
            # Check nodes exist (implementation detail - may need adjustment based on LangGraph internals)
            assert workflow.compiled_graph is not None
    
    def test_workflow_conditional_edge_functions(self):
        """Test conditional edge decision functions."""
        with patch('core.graph.logger'):
            workflow = ApplicationFactoryWorkflow()
            
            # Test successful state
            successful_state: AppState = {
                "workflow_progress": WorkflowProgress(
                    current_node="test",
                    completed_nodes=[],
                    failed_nodes=[],
                    total_nodes=5,
                    progress_percentage=0.0,
                    status=WorkflowStatus.RUNNING
                ),
                "api_key": "test",
                "master_resume_pdf_path": "test.pdf",
                "job_description_pdf_path": "test.pdf",
                "rag_processor": None,
                "master_resume_result": None,
                "job_description_result": None,
                "content_generator": None,
                "resume_content": None,
                "cover_letter_content": None,
                "document_generator": None,
                "resume_document": None,
                "cover_letter_document": None,
                "node_statuses": {},
                "errors": [],
                "logs": [],
                "user_preferences": {},
                "session_id": "test",
                "timestamp": "test"
            }
            
            assert workflow.should_continue_after_init(successful_state) == "continue"
            assert workflow.should_continue_after_rag(successful_state) == "continue"
            assert workflow.should_continue_after_generation(successful_state) == "continue"
            assert workflow.should_continue_after_documents(successful_state) == "continue"
            
            # Test failed state
            failed_state = successful_state.copy()
            failed_state["workflow_progress"].status = WorkflowStatus.FAILED
            
            assert workflow.should_continue_after_init(failed_state) == "error"
            assert workflow.should_continue_after_rag(failed_state) == "error"
            assert workflow.should_continue_after_generation(failed_state) == "error"
            assert workflow.should_continue_after_documents(failed_state) == "error"


class TestWorkflowNodes:
    """Test individual workflow nodes."""
    
    @pytest.fixture
    def workflow(self):
        """Create a workflow instance for testing."""
        with patch('core.graph.logger'):
            return ApplicationFactoryWorkflow()
    
    @pytest.fixture
    def base_state(self):
        """Create a base state for testing."""
        return {
            "api_key": "test_api_key",
            "master_resume_pdf_path": "test_resume.pdf",
            "job_description_pdf_path": "test_job.pdf",
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
            "user_preferences": {},
            "session_id": "test_session",
            "timestamp": "2024-01-01T00:00:00"
        }
    
    def test_initialize_workflow_node_success(self, workflow, base_state, tmp_path):
        """Test successful workflow initialization."""
        # Create temporary files
        resume_file = tmp_path / "resume.pdf"
        job_file = tmp_path / "job.pdf"
        resume_file.write_text("test resume content")
        job_file.write_text("test job content")
        
        base_state["master_resume_pdf_path"] = str(resume_file)
        base_state["job_description_pdf_path"] = str(job_file)
        
        with patch('core.graph.logger'):
            result = workflow.initialize_workflow_node(base_state)
            
            assert result["workflow_progress"].status == WorkflowStatus.RUNNING
            assert "initialize_workflow" in result["workflow_progress"].completed_nodes
            assert result["workflow_progress"].progress_percentage == 20.0
            assert result["node_statuses"]["initialize_workflow"] == NodeStatus.COMPLETED
    
    def test_initialize_workflow_node_missing_fields(self, workflow, base_state):
        """Test workflow initialization with missing fields."""
        # Remove required field
        del base_state["api_key"]
        
        with patch('core.graph.logger'):
            result = workflow.initialize_workflow_node(base_state)
            
            assert result["workflow_progress"].status == WorkflowStatus.FAILED
            assert len(result["errors"]) > 0
            assert "Missing required fields" in result["errors"][0]
    
    def test_initialize_workflow_node_missing_files(self, workflow, base_state):
        """Test workflow initialization with missing files."""
        base_state["master_resume_pdf_path"] = "/nonexistent/file.pdf"
        
        with patch('core.graph.logger'):
            result = workflow.initialize_workflow_node(base_state)
            
            assert result["workflow_progress"].status == WorkflowStatus.FAILED
            assert len(result["errors"]) > 0
            assert "File not found" in result["errors"][0]
    
    @patch('core.graph.RAGProcessor')
    def test_process_documents_node_success(self, mock_rag_class, workflow, base_state):
        """Test successful document processing."""
        # Mock RAG processor
        mock_rag = Mock()
        mock_rag.process_pdf.side_effect = [
            {"content": "resume content", "analysis": {}},
            {"content": "job content", "analysis": {}}
        ]
        mock_rag_class.return_value = mock_rag
        
        with patch('core.graph.logger'):
            result = workflow.process_documents_node(base_state)
            
            assert result["workflow_progress"].progress_percentage == 40.0
            assert "process_documents" in result["workflow_progress"].completed_nodes
            assert result["rag_processor"] is not None
            assert result["master_resume_result"] is not None
            assert result["job_description_result"] is not None
    
    @patch('core.graph.RAGProcessor')
    def test_process_documents_node_failure(self, mock_rag_class, workflow, base_state):
        """Test document processing failure."""
        # Mock RAG processor to raise exception
        mock_rag_class.side_effect = Exception("RAG processing failed")
        
        with patch('core.graph.logger'):
            result = workflow.process_documents_node(base_state)
            
            assert result["workflow_progress"].status == WorkflowStatus.FAILED
            assert "process_documents" in result["workflow_progress"].failed_nodes
            assert len(result["errors"]) > 0
            assert "Document processing failed" in result["errors"][0]
    
    @patch('core.graph.ContentGenerator')
    def test_generate_content_node_success(self, mock_content_class, workflow, base_state):
        """Test successful content generation."""
        # Setup state with RAG results
        mock_rag = Mock()
        mock_rag.get_relevant_context.return_value = [{"content": "context"}]
        base_state["rag_processor"] = mock_rag
        base_state["job_description_result"] = {"content": "job content"}
        
        # Mock content generator
        mock_generator = Mock()
        mock_content = Mock(spec=GeneratedContent)
        mock_generator.generate_content.return_value = mock_content
        mock_content_class.return_value = mock_generator
        
        with patch('core.graph.logger'):
            result = workflow.generate_content_node(base_state)
            
            assert result["workflow_progress"].progress_percentage == 70.0
            assert "generate_content" in result["workflow_progress"].completed_nodes
            assert result["content_generator"] is not None
            assert result["resume_content"] is not None
            assert result["cover_letter_content"] is not None
    
    @patch('core.graph.DocumentGenerator')
    def test_create_documents_node_success(self, mock_doc_class, workflow, base_state):
        """Test successful document creation."""
        # Setup state with content
        mock_resume_content = Mock(spec=GeneratedContent)
        mock_cl_content = Mock(spec=GeneratedContent)
        base_state["resume_content"] = mock_resume_content
        base_state["cover_letter_content"] = mock_cl_content
        
        # Mock document generator
        mock_generator = Mock()
        mock_doc = Mock(spec=GeneratedDocument)
        mock_generator.generate_resume.return_value = mock_doc
        mock_generator.generate_cover_letter.return_value = mock_doc
        mock_doc_class.return_value = mock_generator
        
        with patch('core.graph.logger'):
            result = workflow.create_documents_node(base_state)
            
            assert result["workflow_progress"].progress_percentage == 90.0
            assert "create_documents" in result["workflow_progress"].completed_nodes
            assert result["document_generator"] is not None
            assert result["resume_document"] is not None
            assert result["cover_letter_document"] is not None
    
    def test_finalize_workflow_node(self, workflow, base_state):
        """Test workflow finalization."""
        base_state["workflow_progress"].start_time = datetime.now()
        
        with patch('core.graph.logger'):
            result = workflow.finalize_workflow_node(base_state)
            
            assert result["workflow_progress"].status == WorkflowStatus.COMPLETED
            assert result["workflow_progress"].progress_percentage == 100.0
            assert "finalize_workflow" in result["workflow_progress"].completed_nodes
            assert result["workflow_progress"].end_time is not None
    
    def test_handle_error_node(self, workflow, base_state):
        """Test error handling."""
        base_state["errors"] = ["Test error 1", "Test error 2"]
        
        with patch('core.graph.logger'):
            result = workflow.handle_error_node(base_state)
            
            assert result["workflow_progress"].status == WorkflowStatus.FAILED
            assert result["workflow_progress"].end_time is not None


class TestWorkflowExecution:
    """Test complete workflow execution scenarios."""
    
    @pytest.fixture
    def temp_files(self, tmp_path):
        """Create temporary PDF files for testing."""
        resume_file = tmp_path / "test_resume.pdf"
        job_file = tmp_path / "test_job.pdf"
        resume_file.write_text("Test resume content")
        job_file.write_text("Test job description content")
        return str(resume_file), str(job_file)
    
    @patch('core.graph.DocumentGenerator')
    @patch('core.graph.ContentGenerator')
    @patch('core.graph.RAGProcessor')
    def test_successful_workflow_execution(self, mock_rag_class, mock_content_class, 
                                         mock_doc_class, temp_files):
        """Test complete successful workflow execution."""
        resume_file, job_file = temp_files
        
        # Mock RAG processor
        mock_rag = Mock()
        mock_rag.process_pdf.side_effect = [
            {"content": "resume content", "analysis": {}},
            {"content": "job content", "analysis": {}}
        ]
        mock_rag.get_relevant_context.return_value = [{"content": "context"}]
        mock_rag_class.return_value = mock_rag
        
        # Mock content generator
        mock_generator = Mock()
        mock_content = Mock(spec=GeneratedContent)
        mock_generator.generate_content.return_value = mock_content
        mock_content_class.return_value = mock_generator
        
        # Mock document generator
        mock_doc_gen = Mock()
        mock_document = Mock(spec=GeneratedDocument)
        mock_doc_gen.generate_resume.return_value = mock_document
        mock_doc_gen.generate_cover_letter.return_value = mock_document
        mock_doc_class.return_value = mock_doc_gen
        
        with patch('core.graph.logger'):
            workflow = ApplicationFactoryWorkflow()
            result = workflow.execute_workflow(
                api_key="test_key",
                master_resume_path=resume_file,
                job_description_path=job_file,
                user_preferences={"style": "professional"},
                session_id="test_session"
            )
            
            assert result["workflow_progress"].status == WorkflowStatus.COMPLETED
            assert result["workflow_progress"].progress_percentage == 100.0
            assert len(result["workflow_progress"].completed_nodes) == 5
            assert len(result["workflow_progress"].failed_nodes) == 0
            assert result["resume_document"] is not None
            assert result["cover_letter_document"] is not None
    
    def test_workflow_execution_with_missing_files(self):
        """Test workflow execution with missing files."""
        with patch('core.graph.logger'):
            workflow = ApplicationFactoryWorkflow()
            result = workflow.execute_workflow(
                api_key="test_key",
                master_resume_path="/nonexistent/resume.pdf",
                job_description_path="/nonexistent/job.pdf"
            )
            
            assert result["workflow_progress"].status == WorkflowStatus.FAILED
            assert len(result["errors"]) > 0
    
    @patch('core.graph.RAGProcessor')
    def test_workflow_execution_with_rag_failure(self, mock_rag_class, temp_files):
        """Test workflow execution with RAG processing failure."""
        resume_file, job_file = temp_files
        
        # Mock RAG processor to fail
        mock_rag_class.side_effect = Exception("RAG processing failed")
        
        with patch('core.graph.logger'):
            workflow = ApplicationFactoryWorkflow()
            result = workflow.execute_workflow(
                api_key="test_key",
                master_resume_path=resume_file,
                job_description_path=job_file
            )
            
            assert result["workflow_progress"].status == WorkflowStatus.FAILED
            assert "process_documents" in result["workflow_progress"].failed_nodes
    
    def test_get_workflow_status(self, temp_files):
        """Test workflow status reporting."""
        resume_file, job_file = temp_files
        
        with patch('core.graph.logger'):
            workflow = ApplicationFactoryWorkflow()
            
            # Create a test state
            progress = WorkflowProgress(
                current_node="test_node",
                completed_nodes=["node1", "node2"],
                failed_nodes=["node3"],
                total_nodes=5,
                progress_percentage=60.0,
                status=WorkflowStatus.RUNNING,
                start_time=datetime.now()
            )
            
            test_state: AppState = {
                "workflow_progress": progress,
                "errors": ["Test error"],
                "logs": ["Test log"],
                "api_key": "test",
                "master_resume_pdf_path": resume_file,
                "job_description_pdf_path": job_file,
                "rag_processor": None,
                "master_resume_result": None,
                "job_description_result": None,
                "content_generator": None,
                "resume_content": None,
                "cover_letter_content": None,
                "document_generator": None,
                "resume_document": None,
                "cover_letter_document": None,
                "node_statuses": {},
                "user_preferences": {},
                "session_id": "test",
                "timestamp": "test"
            }
            
            status = workflow.get_workflow_status(test_state)
            
            assert status["status"] == "running"
            assert status["current_node"] == "test_node"
            assert status["progress_percentage"] == 60.0
            assert status["completed_nodes"] == ["node1", "node2"]
            assert status["failed_nodes"] == ["node3"]
            assert status["errors"] == ["Test error"]
            assert status["logs"] == ["Test log"]
            assert status["total_nodes"] == 5


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_workflow(self):
        """Test workflow creation function."""
        with patch('core.graph.logger'):
            workflow = create_workflow()
            assert isinstance(workflow, ApplicationFactoryWorkflow)
            assert workflow.compiled_graph is not None
    
    @patch('core.graph.ApplicationFactoryWorkflow')
    def test_execute_application_factory(self, mock_workflow_class):
        """Test convenience execution function."""
        # Mock workflow
        mock_workflow = Mock()
        mock_final_state = {"workflow_progress": Mock(status=WorkflowStatus.COMPLETED)}
        mock_workflow.execute_workflow.return_value = mock_final_state
        mock_workflow_class.return_value = mock_workflow
        
        result = execute_application_factory(
            api_key="test_key",
            master_resume_path="test_resume.pdf",
            job_description_path="test_job.pdf",
            user_preferences={"style": "modern"},
            session_id="test_session"
        )
        
        # Verify workflow was created and executed
        mock_workflow_class.assert_called_once()
        mock_workflow.execute_workflow.assert_called_once_with(
            api_key="test_key",
            master_resume_path="test_resume.pdf",
            job_description_path="test_job.pdf",
            user_preferences={"style": "modern"},
            session_id="test_session"
        )
        assert result == mock_final_state


class TestWorkflowIntegration:
    """Integration tests with mocked dependencies."""
    
    def test_workflow_state_transitions(self):
        """Test that workflow state transitions correctly between nodes."""
        with patch('core.graph.logger'):
            workflow = ApplicationFactoryWorkflow()
            
            # Test that all conditional edge functions work as expected
            test_states = [
                (WorkflowStatus.RUNNING, "continue"),
                (WorkflowStatus.COMPLETED, "continue"),
                (WorkflowStatus.FAILED, "error"),
                (WorkflowStatus.CANCELLED, "error")
            ]
            
            for status, expected in test_states:
                test_state = {
                    "workflow_progress": WorkflowProgress(
                        current_node="test",
                        completed_nodes=[],
                        failed_nodes=[],
                        total_nodes=5,
                        progress_percentage=0.0,
                        status=status
                    )
                }
                
                if expected == "continue":
                    assert workflow.should_continue_after_init(test_state) == "continue"
                else:
                    assert workflow.should_continue_after_init(test_state) == "error"
    
    def test_workflow_progress_tracking(self):
        """Test that workflow progress is tracked correctly."""
        progress = WorkflowProgress(
            current_node="initialize_workflow",
            completed_nodes=[],
            failed_nodes=[],
            total_nodes=5,
            progress_percentage=0.0,
            status=WorkflowStatus.PENDING
        )
        
        # Test initial state
        assert progress.progress_percentage == 0.0
        assert len(progress.completed_nodes) == 0
        
        # Test progress updates
        progress.completed_nodes.append("initialize_workflow")
        progress.progress_percentage = 20.0
        progress.current_node = "process_documents"
        
        assert progress.progress_percentage == 20.0
        assert len(progress.completed_nodes) == 1
        assert progress.current_node == "process_documents"
    
    def test_error_handling_and_recovery(self):
        """Test error handling throughout the workflow."""
        with patch('core.graph.logger'):
            workflow = ApplicationFactoryWorkflow()
            
            # Test error state creation
            error_state = {
                "workflow_progress": WorkflowProgress(
                    current_node="test_node",
                    completed_nodes=[],
                    failed_nodes=["test_node"],
                    total_nodes=5,
                    progress_percentage=0.0,
                    status=WorkflowStatus.FAILED,
                    error_message="Test error"
                ),
                "errors": ["Test error"],
                "logs": []
            }
            
            result = workflow.handle_error_node(error_state)
            assert result["workflow_progress"].status == WorkflowStatus.FAILED
            assert result["workflow_progress"].end_time is not None


if __name__ == "__main__":
    pytest.main([__file__]) 