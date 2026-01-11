"""
Generator Factory - Creates LLM generators by provider name

Supports: gemini, groq, deepseek
"""

from typing import Optional

from ccba_rag.core.base import BaseGenerator
from ccba_rag.core.settings import settings
from ccba_rag.utils.logging import get_logger

logger = get_logger(__name__)


def create_generator(
    provider: str = "gemini",
    **kwargs
) -> BaseGenerator:
    """
    Create a generator instance by provider name.
    
    Args:
        provider: One of 'gemini', 'groq', 'deepseek'
        **kwargs: Additional arguments passed to generator constructor
        
    Returns:
        Configured generator instance
    """
    provider = provider.lower()
    
    if provider == "gemini":
        from ccba_rag.generation.generators.gemini import GeminiGenerator
        return GeminiGenerator(**kwargs)
    
    elif provider == "groq":
        from ccba_rag.generation.generators.groq import GroqGenerator
        return GroqGenerator(**kwargs)
    
    elif provider == "deepseek":
        from ccba_rag.generation.generators.deepseek import DeepSeekGenerator
        return DeepSeekGenerator(**kwargs)
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: gemini, groq, deepseek")


def create_fallback_generator(
    primary_provider: str = "gemini",
    fallback_provider: str = "groq"
) -> tuple:
    """
    Create primary and fallback generator pair.
    
    Args:
        primary_provider: Primary generator provider
        fallback_provider: Fallback generator provider
        
    Returns:
        Tuple of (primary_generator, fallback_generator)
    """
    primary = create_generator(primary_provider)
    
    try:
        fallback = create_generator(fallback_provider)
    except ValueError:
        logger.warning(f"Failed to create fallback generator ({fallback_provider})")
        fallback = None
    
    return primary, fallback


class GeneratorFactory:
    """
    Factory class for creating and caching generators.
    """
    
    _instances = {}
    
    @classmethod
    def get(cls, provider: str = "gemini") -> BaseGenerator:
        """
        Get or create a generator instance (cached).
        
        Args:
            provider: Provider name
            
        Returns:
            Generator instance
        """
        if provider not in cls._instances:
            cls._instances[provider] = create_generator(provider)
        return cls._instances[provider]
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached generator instances."""
        cls._instances.clear()
