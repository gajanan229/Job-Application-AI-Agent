"""Tests for the configuration system."""

import pytest
import os
from unittest.mock import patch
from config.settings import AppConfig


class TestAppConfig:
    """Test class for AppConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AppConfig()
        
        assert config.gemini_model == "gemini-pro"
        assert config.embedding_model == "models/embedding-001"
        assert config.app_env == "development"
        assert config.debug is True
        assert config.max_file_size_mb == 10
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid max_file_size_mb
        with pytest.raises(ValueError, match="max_file_size_mb must be positive"):
            AppConfig(max_file_size_mb=0)
        
        # Test invalid chunk_size
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            AppConfig(chunk_size=0)
        
        # Test invalid chunk_overlap
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            AppConfig(chunk_overlap=-1)
        
        # Test chunk_overlap >= chunk_size
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            AppConfig(chunk_size=100, chunk_overlap=100)
        
        # Test invalid temperature
        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            AppConfig(temperature=-1)
        
        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            AppConfig(temperature=3)
        
        # Test invalid max_tokens
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            AppConfig(max_tokens=0)
    
    @patch.dict(os.environ, {
        'GOOGLE_API_KEY': 'test-key-123',
        'GEMINI_MODEL': 'gemini-pro-test',
        'MAX_FILE_SIZE_MB': '5',
        'DEBUG': 'false'
    })
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        config = AppConfig.load_from_env()
        
        assert config.google_api_key == 'test-key-123'
        assert config.gemini_model == 'gemini-pro-test'
        assert config.max_file_size_mb == 5
        assert config.debug is False
    
    def test_is_api_key_configured(self):
        """Test API key configuration check."""
        # No API key
        config = AppConfig(google_api_key=None)
        assert not config.is_api_key_configured()
        
        # Empty API key
        config = AppConfig(google_api_key="")
        assert not config.is_api_key_configured()
        
        # Valid API key
        config = AppConfig(google_api_key="test-key-123")
        assert config.is_api_key_configured()
    
    def test_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = AppConfig(
            google_api_key="test-key",
            gemini_model="test-model"
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['google_api_key'] == "test-key"
        assert config_dict['gemini_model'] == "test-model"
        assert 'max_file_size_mb' in config_dict 