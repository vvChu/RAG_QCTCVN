"""
Test Settings Module

Tests configuration loading from:
- Environment variables
- YAML files
- Default values
"""

import pytest
import os


class TestSettings:
    """Tests for settings module."""
    
    def test_settings_import(self):
        """Test that settings can be imported."""
        from ccba_rag.core.settings import settings
        assert settings is not None
    
    def test_settings_has_milvus_config(self):
        """Test that Milvus settings are present."""
        from ccba_rag.core.settings import settings
        assert hasattr(settings, 'milvus_host')
        assert hasattr(settings, 'milvus_port')
        assert hasattr(settings, 'milvus_collection_name')
    
    def test_settings_has_bge_config(self):
        """Test that BGE-M3 settings are present."""
        from ccba_rag.core.settings import settings
        assert hasattr(settings, 'bge_model_name')
        assert settings.bge_model_name == "BAAI/bge-m3"
        assert settings.bge_max_length == 1024
    
    def test_settings_has_llm_config(self):
        """Test that LLM settings are present."""
        from ccba_rag.core.settings import settings
        assert hasattr(settings, 'gemini_model')
        assert hasattr(settings, 'groq_model')
    
    def test_get_prompts(self):
        """Test that prompts can be loaded."""
        from ccba_rag.core.settings import settings
        prompts = settings.get_prompts()
        assert isinstance(prompts, dict)


class TestPromptManager:
    """Tests for prompt management."""
    
    def test_prompt_manager_import(self):
        """Test that prompt manager can be imported."""
        from ccba_rag.core.prompts import prompt_manager
        assert prompt_manager is not None
    
    def test_system_instruction_exists(self):
        """Test that system instruction is loaded."""
        from ccba_rag.core.prompts import prompt_manager
        instruction = prompt_manager.system_instruction
        assert instruction is not None
        assert len(instruction) > 100  # Should be substantial


class TestModels:
    """Tests for Pydantic models."""
    
    def test_chunk_model(self):
        """Test Chunk model creation."""
        from ccba_rag.core.models import Chunk
        
        chunk = Chunk(
            id="test-id",
            text="Test content",
            document_id="doc-1",
            document_name="test.pdf"
        )
        assert chunk.id == "test-id"
        assert chunk.text == "Test content"
    
    def test_rag_response_model(self):
        """Test RAGResponse model creation."""
        from ccba_rag.core.models import RAGResponse
        
        response = RAGResponse(
            answer="Test answer"
        )
        assert response.answer == "Test answer"
        assert response.citations == []
        assert response.contexts == []
