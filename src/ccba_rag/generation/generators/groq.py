"""
Groq Generator for RAG Answer Generation

Uses Groq API for fast LLM inference with Llama models.
Typically used as fallback when Gemini is rate-limited.
"""

from typing import Any, Dict, List, Optional

from ccba_rag.core.base import BaseGenerator
from ccba_rag.core.prompts import prompt_manager
from ccba_rag.core.settings import settings
from ccba_rag.utils.logging import get_logger

logger = get_logger(__name__)


class GroqGenerator(BaseGenerator):
    """
    Groq LLM generator using Llama models.

    Fast inference, good for fallback scenarios.
    Default model: llama-3.3-70b-versatile
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize Groq Generator.

        Args:
            api_key: Groq API key (or from GROQ_API_KEY env)
            model: Model name (default: llama-3.3-70b-versatile)
        """
        self.api_key = api_key or settings.groq_api_key
        self._model = model or settings.groq_model

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found. Set in .env or pass as argument.")

        self._client = None
        logger.info(f"GroqGenerator initialized with model: {self._model}")

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model

    @property
    def client(self):
        """Lazy-load Groq client."""
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

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
        Generate answer from Groq.

        Args:
            query: User question
            contexts: Retrieved context documents
            history: Optional conversation history
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dict with keys: answer, model, prompt, error, error_type
        """
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
            # Build messages for chat API
            messages = [
                {"role": "system", "content": prompt_manager.system_instruction},
                {"role": "user", "content": full_prompt}
            ]

            response = self.client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            result_text = response.choices[0].message.content
            return {**base_result, 'answer': result_text}

        except Exception as e:
            error_type = self._classify_error(e)
            logger.error(f"Groq generation failed ({error_type}): {e}")
            return {**base_result, 'error': str(e), 'error_type': error_type}

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 1024
    ) -> str:
        """Direct text generation without RAG template."""
        messages = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    def _classify_error(self, error: Exception) -> str:
        """Classify error type for reporting."""
        error_str = str(error).lower()

        if 'rate' in error_str or 'limit' in error_str:
            return 'RATE_LIMIT_ERROR'
        elif 'unavailable' in error_str or 'timeout' in error_str:
            return 'API_UNAVAILABLE'
        elif 'authentication' in error_str or 'api_key' in error_str:
            return 'AUTH_ERROR'
        else:
            return 'UNKNOWN_ERROR'
