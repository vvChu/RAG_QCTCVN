"""
Gemini Generator for RAG Answer Generation

Uses Google Generative AI SDK for LLM inference.
Supports conversation history and structured error handling.
"""

from typing import List, Dict, Optional, Any

from ccba_rag.core.base import BaseGenerator
from ccba_rag.core.settings import settings
from ccba_rag.core.prompts import prompt_manager
from ccba_rag.utils.logging import get_logger

logger = get_logger(__name__)


class GeminiGenerator(BaseGenerator):
    """
    Gemini LLM generator for RAG answer generation.
    
    Features:
    - Retrieval-first (R+G) strategy
    - Citation formatting
    - Conversation history support
    - Structured error handling for fallback
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize Gemini Generator.
        
        Args:
            api_key: Gemini API key (or from GEMINI_API_KEY env)
            model: Model name (default: gemini-2.0-flash)
        """
        self.api_key = api_key or settings.gemini_api_key
        self._model = model or settings.gemini_model
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Set in .env or pass as argument.")
        
        logger.info(f"GeminiGenerator initialized with model: {self._model}")
    
    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model
    
    def format_contexts(self, contexts: List[Dict]) -> str:
        """Format contexts using the prompt manager."""
        return prompt_manager.format_contexts(contexts)
    
    def generate(
        self,
        query: str,
        contexts: List[Dict],
        history: Optional[List[Dict]] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Generate answer from Gemini.
        
        Args:
            query: User question
            contexts: Retrieved context documents
            history: Optional conversation history
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with keys: answer, model, prompt, error, error_type
        """
        # Build prompt
        full_prompt = prompt_manager.build_rag_prompt(query, contexts, history)
        
        base_result = {
            'answer': None,
            'contexts_used': contexts,
            'prompt': full_prompt,
            'model': self._model,
            'error': None,
            'error_type': None,
        }
        
        try:
            result_text = self._call_gemini(
                prompt=full_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {**base_result, 'answer': result_text}
        
        except Exception as e:
            error_type = self._classify_error(e)
            logger.error(f"Gemini generation failed ({error_type}): {e}")
            return {**base_result, 'error': str(e), 'error_type': error_type}
    
    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 1024
    ) -> str:
        """Direct text generation without RAG template."""
        return self._call_gemini(prompt, temperature, max_tokens)
    
    def _call_gemini(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        Call Gemini API using google-generativeai SDK.
        """
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            model_instance = genai.GenerativeModel(self._model)
            
            response = model_instance.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            
            return response.text
        
        except ImportError:
            raise RuntimeError("google-generativeai not installed. Run: pip install google-generativeai")
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error type for fallback logic."""
        error_str = str(error).lower()
        
        if 'quota' in error_str or 'rate' in error_str or 'exhausted' in error_str:
            return 'RATE_LIMIT_ERROR'
        elif 'unavailable' in error_str or 'timeout' in error_str:
            return 'API_UNAVAILABLE'
        elif 'invalid' in error_str or 'api_key' in error_str:
            return 'AUTH_ERROR'
        else:
            return 'UNKNOWN_ERROR'
