"""
Tests for LLM Service Module

This module tests the LLM integration, prompt management, and content generation
functionality of the Application Factory.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, List, Any

from core.llm_service import (
    PromptManager, LLMService, ContentGenerator,
    ContentType, GenerationRequest, GenerationResponse
)
from utils.error_handlers import LLMError, ValidationError


class TestPromptManager:
    """Test cases for PromptManager class."""
    
    def test_initialization(self):
        """Test PromptManager initialization."""
        manager = PromptManager()
        assert hasattr(manager, '_prompts')
        assert isinstance(manager._prompts, dict)
        
        # Check that all expected prompts are initialized
        expected_prompts = [
            "resume_system", "resume_user",
            "cover_letter_system", "cover_letter_user",
            "analysis_system", "analysis_user"
        ]
        for prompt_key in expected_prompts:
            assert prompt_key in manager._prompts
            assert isinstance(manager._prompts[prompt_key], str)
            assert len(manager._prompts[prompt_key]) > 0
    
    def test_get_prompt_success(self):
        """Test successful prompt retrieval."""
        manager = PromptManager()
        
        # Test getting system prompt without parameters
        resume_system = manager.get_prompt("resume_system")
        assert isinstance(resume_system, str)
        assert "resume writer" in resume_system.lower()
        
        # Test getting user prompt with parameters
        resume_user = manager.get_prompt(
            "resume_user",
            job_description="Software Engineer position",
            master_resume_text="John Doe Resume",
            rag_context="Relevant context",
            additional_instructions="Custom instructions"
        )
        assert isinstance(resume_user, str)
        assert "Software Engineer position" in resume_user
        assert "John Doe Resume" in resume_user
    
    def test_get_prompt_unknown_type(self):
        """Test getting prompt with unknown type."""
        manager = PromptManager()
        
        with pytest.raises(ValidationError):
            manager.get_prompt("unknown_prompt_type")
    
    def test_get_prompt_missing_parameter(self):
        """Test getting prompt with missing required parameter."""
        manager = PromptManager()
        
        with pytest.raises(ValidationError):
            manager.get_prompt(
                "resume_user",
                job_description="Software Engineer"
                # Missing other required parameters
            )
    
    def test_all_prompt_types(self):
        """Test that all prompt types can be retrieved."""
        manager = PromptManager()
        
        prompt_types = [
            "resume_system", "cover_letter_system", "analysis_system"
        ]
        
        for prompt_type in prompt_types:
            prompt = manager.get_prompt(prompt_type)
            assert isinstance(prompt, str)
            assert len(prompt) > 100  # Reasonable length for system prompts


class TestLLMService:
    """Test cases for LLMService class."""
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_initialization_success(self, mock_model, mock_configure):
        """Test successful LLM service initialization."""
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        service = LLMService(api_key="test_key")
        
        assert service.api_key == "test_key"
        assert service.model == mock_model_instance
        assert isinstance(service.prompt_manager, PromptManager)
        
        mock_configure.assert_called_once_with(api_key="test_key")
        mock_model.assert_called_once()
    
    @patch('config.settings.config.google_api_key', None)
    def test_initialization_no_api_key(self):
        """Test initialization without API key."""
        with pytest.raises(LLMError):
            LLMService()
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_initialization_api_failure(self, mock_model, mock_configure):
        """Test initialization with API failure."""
        mock_configure.side_effect = Exception("API configuration failed")
        
        with pytest.raises(LLMError):
            LLMService(api_key="test_key")
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_generate_content_success(self, mock_model, mock_configure):
        """Test successful content generation."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.text = "Generated resume content"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 200
        mock_response.usage_metadata.total_token_count = 300
        
        mock_model_instance.generate_content.return_value = mock_response
        mock_model.return_value = mock_model_instance
        
        service = LLMService(api_key="test_key")
        
        # Create request
        request = GenerationRequest(
            content_type=ContentType.RESUME,
            rag_context=[{
                'content': 'Test context',
                'similarity_score': 0.8,
                'source': 'test_doc'
            }],
            job_description="Software Engineer position",
            master_resume_text="John Doe Resume"
        )
        
        # Generate content
        response = service.generate_content(request)
        
        # Assertions
        assert isinstance(response, GenerationResponse)
        assert response.content == "Generated resume content"
        assert response.content_type == ContentType.RESUME
        assert response.generation_time > 0
        assert response.token_usage is not None
        assert response.token_usage['total_tokens'] == 300
        
        mock_model_instance.generate_content.assert_called_once()
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_generate_content_empty_response(self, mock_model, mock_configure):
        """Test content generation with empty response."""
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.text = ""  # Empty response
        
        mock_model_instance.generate_content.return_value = mock_response
        mock_model.return_value = mock_model_instance
        
        service = LLMService(api_key="test_key")
        
        request = GenerationRequest(
            content_type=ContentType.RESUME,
            rag_context=[],
            job_description="Test job",
            master_resume_text="Test resume"
        )
        
        with pytest.raises(LLMError):
            service.generate_content(request)
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_generate_content_api_failure(self, mock_model, mock_configure):
        """Test content generation with API failure."""
        mock_model_instance = Mock()
        mock_model_instance.generate_content.side_effect = Exception("API call failed")
        mock_model.return_value = mock_model_instance
        
        service = LLMService(api_key="test_key")
        
        request = GenerationRequest(
            content_type=ContentType.RESUME,
            rag_context=[],
            job_description="Test job",
            master_resume_text="Test resume"
        )
        
        with pytest.raises(LLMError):
            service.generate_content(request)
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_format_rag_context(self, mock_model, mock_configure):
        """Test RAG context formatting."""
        mock_model.return_value = Mock()
        service = LLMService(api_key="test_key")
        
        # Test with empty context
        empty_context = service._format_rag_context([])
        assert "No additional context available" in empty_context
        
        # Test with context data
        rag_context = [
            {
                'content': 'First chunk of content',
                'similarity_score': 0.9,
                'source': 'resume.pdf'
            },
            {
                'content': 'Second chunk of content',
                'similarity_score': 0.8,
                'source': 'job_description.pdf'
            }
        ]
        
        formatted = service._format_rag_context(rag_context)
        assert "First chunk of content" in formatted
        assert "0.900" in formatted
        assert "resume.pdf" in formatted
        assert "Second chunk of content" in formatted
        assert "0.800" in formatted
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_extract_token_usage(self, mock_model, mock_configure):
        """Test token usage extraction."""
        mock_model.return_value = Mock()
        service = LLMService(api_key="test_key")
        
        # Test with usage metadata
        mock_response = Mock()
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 200
        mock_response.usage_metadata.total_token_count = 300
        
        usage = service._extract_token_usage(mock_response)
        assert usage is not None
        assert usage['prompt_tokens'] == 100
        assert usage['completion_tokens'] == 200
        assert usage['total_tokens'] == 300
        
        # Test without usage metadata
        mock_response_no_usage = Mock()
        del mock_response_no_usage.usage_metadata  # Simulate no usage metadata
        
        usage = service._extract_token_usage(mock_response_no_usage)
        assert usage is None


class TestContentGenerator:
    """Test cases for ContentGenerator class."""
    
    @patch('core.llm_service.LLMService')
    def test_initialization(self, mock_llm_service):
        """Test ContentGenerator initialization."""
        generator = ContentGenerator(api_key="test_key")
        
        assert hasattr(generator, 'llm_service')
        mock_llm_service.assert_called_once_with("test_key")
    
    @patch('core.llm_service.LLMService')
    def test_generate_resume(self, mock_llm_service):
        """Test resume generation."""
        # Setup mock
        mock_service_instance = Mock()
        mock_response = GenerationResponse(
            content="Generated resume",
            content_type=ContentType.RESUME,
            metadata={},
            generation_time=1.5
        )
        mock_service_instance.generate_content.return_value = mock_response
        mock_llm_service.return_value = mock_service_instance
        
        generator = ContentGenerator(api_key="test_key")
        
        # Generate resume
        result = generator.generate_resume(
            job_description="Software Engineer",
            master_resume_text="John Doe Resume",
            rag_context=[{'content': 'test'}],
            user_preferences={'style': 'modern'}
        )
        
        # Assertions
        assert result == mock_response
        mock_service_instance.generate_content.assert_called_once()
        
        # Check the request passed to LLM service
        call_args = mock_service_instance.generate_content.call_args[0][0]
        assert call_args.content_type == ContentType.RESUME
        assert call_args.job_description == "Software Engineer"
        assert call_args.master_resume_text == "John Doe Resume"
        assert call_args.user_preferences == {'style': 'modern'}
    
    @patch('core.llm_service.LLMService')
    def test_generate_cover_letter(self, mock_llm_service):
        """Test cover letter generation."""
        mock_service_instance = Mock()
        mock_response = GenerationResponse(
            content="Generated cover letter",
            content_type=ContentType.COVER_LETTER,
            metadata={},
            generation_time=2.0
        )
        mock_service_instance.generate_content.return_value = mock_response
        mock_llm_service.return_value = mock_service_instance
        
        generator = ContentGenerator(api_key="test_key")
        
        result = generator.generate_cover_letter(
            job_description="Software Engineer",
            master_resume_text="John Doe Resume",
            rag_context=[{'content': 'test'}]
        )
        
        assert result == mock_response
        call_args = mock_service_instance.generate_content.call_args[0][0]
        assert call_args.content_type == ContentType.COVER_LETTER
    
    @patch('core.llm_service.LLMService')
    def test_analyze_documents(self, mock_llm_service):
        """Test document analysis."""
        mock_service_instance = Mock()
        mock_response = GenerationResponse(
            content="Analysis results",
            content_type=ContentType.ANALYSIS,
            metadata={},
            generation_time=1.0
        )
        mock_service_instance.generate_content.return_value = mock_response
        mock_llm_service.return_value = mock_service_instance
        
        generator = ContentGenerator(api_key="test_key")
        
        result = generator.analyze_documents(
            job_description="Software Engineer",
            master_resume_text="John Doe Resume",
            rag_context=[{'content': 'test'}]
        )
        
        assert result == mock_response
        call_args = mock_service_instance.generate_content.call_args[0][0]
        assert call_args.content_type == ContentType.ANALYSIS
    
    @patch('core.llm_service.LLMService')
    def test_batch_generate(self, mock_llm_service):
        """Test batch content generation."""
        mock_service_instance = Mock()
        mock_llm_service.return_value = mock_service_instance
        
        generator = ContentGenerator(api_key="test_key")
        
        # Mock the individual generation methods
        mock_resume_response = Mock()
        mock_cover_letter_response = Mock()
        mock_analysis_response = Mock()
        
        generator.generate_resume = Mock(return_value=mock_resume_response)
        generator.generate_cover_letter = Mock(return_value=mock_cover_letter_response)
        generator.analyze_documents = Mock(return_value=mock_analysis_response)
        
        # Test batch generation
        results = generator.batch_generate(
            job_description="Software Engineer",
            master_resume_text="John Doe Resume",
            rag_context=[{'content': 'test'}],
            content_types=[ContentType.RESUME, ContentType.COVER_LETTER, ContentType.ANALYSIS],
            user_preferences={'style': 'professional'}
        )
        
        # Assertions
        assert len(results) == 3
        assert ContentType.RESUME in results
        assert ContentType.COVER_LETTER in results
        assert ContentType.ANALYSIS in results
        
        # Verify method calls
        generator.generate_resume.assert_called_once_with(
            "Software Engineer", "John Doe Resume", [{'content': 'test'}], {'style': 'professional'}
        )
        generator.generate_cover_letter.assert_called_once_with(
            "Software Engineer", "John Doe Resume", [{'content': 'test'}], {'style': 'professional'}
        )
        generator.analyze_documents.assert_called_once_with(
            "Software Engineer", "John Doe Resume", [{'content': 'test'}]
        )


class TestDataClasses:
    """Test cases for data classes."""
    
    def test_generation_request(self):
        """Test GenerationRequest dataclass."""
        request = GenerationRequest(
            content_type=ContentType.RESUME,
            rag_context=[{'content': 'test'}],
            job_description="Software Engineer",
            master_resume_text="Resume text",
            user_preferences={'style': 'modern'},
            additional_context="Additional info"
        )
        
        assert request.content_type == ContentType.RESUME
        assert request.rag_context == [{'content': 'test'}]
        assert request.job_description == "Software Engineer"
        assert request.master_resume_text == "Resume text"
        assert request.user_preferences == {'style': 'modern'}
        assert request.additional_context == "Additional info"
    
    def test_generation_response(self):
        """Test GenerationResponse dataclass."""
        response = GenerationResponse(
            content="Generated content",
            content_type=ContentType.COVER_LETTER,
            metadata={'model': 'gemini-pro'},
            generation_time=2.5,
            token_usage={'total_tokens': 500}
        )
        
        assert response.content == "Generated content"
        assert response.content_type == ContentType.COVER_LETTER
        assert response.metadata == {'model': 'gemini-pro'}
        assert response.generation_time == 2.5
        assert response.token_usage == {'total_tokens': 500}
    
    def test_content_type_enum(self):
        """Test ContentType enumeration."""
        assert ContentType.RESUME.value == "resume"
        assert ContentType.COVER_LETTER.value == "cover_letter"
        assert ContentType.ANALYSIS.value == "analysis"
        
        # Test that enum values are unique
        values = [content_type.value for content_type in ContentType]
        assert len(values) == len(set(values))