"""
Comprehensive Test Suite for Phase 5: LangGraph Workflow Implementation.

This module contains tests for the LangGraph workflow system, rate limiting,
and orchestrated application generation functionality.
"""

import pytest
import time
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the modules to test
from workflow_graph import (
    RateLimiter, WorkflowMetrics, ApplicationFactoryWorkflow,
    create_workflow, run_application_generation
)
from state_rag import (
    create_initial_state, validate_state_for_llm_generation,
    update_state_stage, set_error_state
)
from llm_utils import LLMManager, create_llm_manager
from rag_utils import RAGManager
from html_pdf_utils import generate_resume_pdf, generate_cover_letter_pdf

# Test data
SAMPLE_MASTER_RESUME = """
Gajanan Vigneswaran
Toronto Metropolitan University, Computer Science
GPA: 4.02/4.33

EXPERIENCE:
Software Developer Intern
TechCorp | Summer 2024
• Developed Python applications using Flask and React
• Implemented machine learning models for data analysis
• Collaborated with cross-functional teams on agile projects

SKILLS:
Programming Languages: Python, JavaScript, Java, C++
Frameworks: React, Flask, Django, Node.js
Tools: Git, Docker, AWS, MongoDB
Soft Skills: Leadership, Problem-solving, Communication

EDUCATION:
Bachelor of Science in Computer Science
Toronto Metropolitan University | 2022-2026
Relevant Coursework: Data Structures, Algorithms, Machine Learning, Database Systems

PROJECTS:
AI-Powered Resume Builder
• Built using Python, React, and OpenAI API
• Implemented RAG system for personalized content generation
• Deployed on AWS with 95% uptime

E-commerce Platform
• Full-stack web application using MERN stack
• Integrated payment processing and user authentication
• Handled 1000+ concurrent users
"""

SAMPLE_JOB_DESCRIPTION = """
Software Engineer - AI/ML
TechStartup Inc.
Toronto, ON

We are seeking a talented Software Engineer to join our AI/ML team. The ideal candidate will have experience with:

REQUIRED SKILLS:
• Python programming and machine learning frameworks
• Experience with React and modern web development
• Knowledge of cloud platforms (AWS, GCP)
• Strong problem-solving and communication skills
• Experience with agile development methodologies

RESPONSIBILITIES:
• Develop and deploy machine learning models
• Build user-facing applications using React
• Collaborate with product and design teams
• Participate in code reviews and technical discussions

QUALIFICATIONS:
• Bachelor's degree in Computer Science or related field
• 2+ years of experience in software development
• Experience with Python, React, and cloud technologies
• Strong analytical and problem-solving skills
"""


class TestRateLimiter:
    """Test the rate limiting functionality."""
    
    def test_rate_limiter_initialization(self):
        """Test RateLimiter initialization."""
        limiter = RateLimiter(max_requests=10, time_window=30)
        assert limiter.max_requests == 10
        assert limiter.time_window == 30
        assert len(limiter.requests) == 0
    
    def test_can_make_request_initially(self):
        """Test that requests can be made initially."""
        limiter = RateLimiter(max_requests=5, time_window=60)
        assert limiter.can_make_request() is True
    
    def test_rate_limit_enforcement(self):
        """Test that rate limiting is enforced."""
        limiter = RateLimiter(max_requests=2, time_window=60)
        
        limiter.record_request()
        assert limiter.can_make_request() is True
        
        limiter.record_request()
        assert limiter.can_make_request() is False
    
    def test_wait_time_calculation(self):
        """Test wait time calculation."""
        limiter = RateLimiter(max_requests=1, time_window=60)
        
        assert limiter.wait_time_until_available() == 0.0
        
        limiter.record_request()
        wait_time = limiter.wait_time_until_available()
        assert 0 < wait_time <= 60


class TestWorkflowMetrics:
    """Test the workflow metrics tracking."""
    
    def test_metrics_initialization(self):
        """Test WorkflowMetrics initialization."""
        metrics = WorkflowMetrics()
        assert metrics.start_time is None
        assert metrics.api_calls == 0
        assert len(metrics.errors) == 0
    
    def test_workflow_timing(self):
        """Test workflow timing."""
        metrics = WorkflowMetrics()
        
        metrics.start_workflow()
        assert metrics.start_time is not None
        
        time.sleep(0.1)
        
        metrics.end_workflow()
        assert metrics.end_time > metrics.start_time
    
    def test_api_call_tracking(self):
        """Test API call tracking."""
        metrics = WorkflowMetrics()
        assert metrics.api_calls == 0
        
        metrics.record_api_call()
        assert metrics.api_calls == 1
    
    def test_error_tracking(self):
        """Test error tracking."""
        metrics = WorkflowMetrics()
        
        metrics.record_error("test_node", "Test error")
        assert len(metrics.errors) == 1
        assert metrics.errors[0]["node"] == "test_node"


class TestApplicationFactoryWorkflow:
    """Test the main workflow class."""
    
    def test_workflow_initialization(self):
        """Test workflow initialization."""
        workflow = ApplicationFactoryWorkflow()
        
        assert workflow.rate_limiter is not None
        assert workflow.metrics is not None
        assert workflow.graph is not None
    
    def test_create_workflow_function(self):
        """Test create_workflow convenience function."""
        workflow = create_workflow()
        assert isinstance(workflow, ApplicationFactoryWorkflow)
        assert workflow.rate_limiter.max_requests == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 