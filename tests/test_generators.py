"""
Test Generator Module

Tests LLM generators (Gemini, Groq, DeepSeek).
"""

import pytest
from unittest.mock import patch, MagicMock


class TestGeneratorFactory:
    """Tests for generator factory."""
    
    def test_create_gemini_generator(self):
        """Test creating Gemini generator."""
        from ccba_rag.generation.factory import create_generator
        
        generator = create_generator("gemini")
        assert generator is not None
        assert "gemini" in generator.model_name.lower()
    
    def test_create_groq_generator(self):
        """Test creating Groq generator."""
        from ccba_rag.generation.factory import create_generator
        
        generator = create_generator("groq")
        assert generator is not None
    
    def test_invalid_provider_raises(self):
        """Test that invalid provider raises error."""
        from ccba_rag.generation.factory import create_generator
        
        with pytest.raises((ValueError, KeyError)):
            create_generator("invalid_provider")
    
    def test_factory_caches_generators(self):
        """Test that factory caches generator instances."""
        from ccba_rag.generation.factory import create_generator, _generator_cache
        
        # Clear cache first
        _generator_cache.clear()
        
        g1 = create_generator("gemini")
        g2 = create_generator("gemini")
        
        assert g1 is g2  # Same cached instance


class TestGeminiGenerator:
    """Tests for Gemini generator."""
    
    def test_gemini_initialization(self):
        """Test Gemini generator initialization."""
        from ccba_rag.generation.generators.gemini import GeminiGenerator
        
        generator = GeminiGenerator()
        assert generator.model_name is not None
    
    def test_gemini_has_generate_method(self):
        """Test that Gemini has generate method."""
        from ccba_rag.generation.generators.gemini import GeminiGenerator
        
        generator = GeminiGenerator()
        assert hasattr(generator, 'generate')
        assert callable(generator.generate)


class TestGroqGenerator:
    """Tests for Groq generator."""
    
    def test_groq_initialization(self):
        """Test Groq generator initialization."""
        from ccba_rag.generation.generators.groq import GroqGenerator
        
        generator = GroqGenerator()
        assert generator.model_name is not None
    
    def test_groq_has_generate_method(self):
        """Test that Groq has generate method."""
        from ccba_rag.generation.generators.groq import GroqGenerator
        
        generator = GroqGenerator()
        assert hasattr(generator, 'generate')
