"""
RAG Chain - Orchestrates retrieval and generation pipeline

Supports fallback between multiple LLM providers.
"""

import time
from typing import Any, Dict, List, Optional

from ccba_rag.core.base import BaseGenerator
from ccba_rag.core.settings import settings
from ccba_rag.utils.logging import get_logger

logger = get_logger(__name__)


class RAGChain:
    """
    End-to-end RAG pipeline: Retrieval -> Generation

    Features:
    - Fallback between primary and secondary generators
    - Faithfulness verification (optional)
    - Detailed statistics tracking
    """

    def __init__(
        self,
        retriever,
        primary_generator: BaseGenerator,
        fallback_generator: Optional[BaseGenerator] = None
    ):
        """
        Initialize RAG Chain.

        Args:
            retriever: HybridRetriever instance
            primary_generator: Primary LLM generator (e.g., Gemini)
            fallback_generator: Fallback generator (e.g., Groq)
        """
        self.retriever = retriever
        self.primary = primary_generator
        self.fallback = fallback_generator

    def query(
        self,
        user_query: str,
        retrieval_query: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        filters: Optional[Dict] = None,
        top_k: int = 100,
        top_n: int = 5,
        temperature: float = 0.1,
        use_reranker: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Execute full RAG query.

        Args:
            user_query: User question (used for generation)
            retrieval_query: Query for retrieval (defaults to user_query)
            history: Conversation history
            filters: Metadata filters
            top_k: Retrieval candidates
            top_n: Final results after reranking
            temperature: Generation temperature
            use_reranker: Override reranker setting

        Returns:
            Dict with answer, contexts, stats, model, etc.
        """
        retrieval_query = retrieval_query or user_query
        start_time = time.time()

        # Stage 1: Retrieval
        contexts, retrieval_stats = self.retriever.retrieve(
            query=retrieval_query,
            filters=filters,
            top_k=top_k,
            top_n=top_n,
            use_reranker=use_reranker
        )

        if not contexts:
            return {
                'answer': "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu.",
                'contexts': [],
                'stats': retrieval_stats,
                'model': None,
                'used_fallback': False,
            }

        # Stage 2: Generation with fallback
        gen_start = time.time()
        gen_result = self.primary.generate(
            query=user_query,
            contexts=contexts,
            history=history,
            temperature=temperature
        )

        used_fallback = False

        # Check if fallback needed
        if self._needs_fallback(gen_result) and self.fallback:
            logger.warning(f"Primary generator failed ({gen_result.get('error_type')}), using fallback")
            gen_result = self.fallback.generate(
                query=user_query,
                contexts=contexts,
                history=history,
                temperature=temperature
            )
            used_fallback = True

        gen_time = (time.time() - gen_start) * 1000
        total_time = (time.time() - start_time) * 1000

        # Build final result
        return {
            'answer': gen_result.get('answer', 'Không thể tạo câu trả lời.'),
            'contexts': contexts,
            'stats': {
                **retrieval_stats,
                'generation_ms': gen_time,
                'total_ms': total_time,
            },
            'model': gen_result.get('model'),
            'prompt': gen_result.get('prompt'),
            'used_fallback': used_fallback,
            'error': gen_result.get('error'),
            'error_type': gen_result.get('error_type'),
        }

    def _needs_fallback(self, gen_result: Dict) -> bool:
        """Check if generation result indicates need for fallback."""
        if gen_result.get('answer'):
            return False

        error_type = gen_result.get('error_type', '')
        # Fallback on rate limits and unavailability
        return error_type in ['RATE_LIMIT_ERROR', 'API_UNAVAILABLE', 'UNKNOWN_ERROR']

    def query_with_verification(
        self,
        user_query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query with faithfulness verification using LLM-as-Judge.

        Returns same as query() plus faithfulness_score and faithfulness_verdict.
        """
        result = self.query(user_query, **kwargs)

        if result.get('answer') and result.get('contexts'):
            verdict, score = self._verify_faithfulness(
                result['answer'],
                result['contexts']
            )
            result['faithfulness_verdict'] = verdict
            result['faithfulness_score'] = score

        return result

    def _verify_faithfulness(
        self,
        answer: str,
        contexts: List[Dict]
    ) -> tuple:
        """
        Verify if answer is supported by contexts using LLM-as-Judge.

        Returns:
            Tuple of (verdict: str, score: float)
        """
        # Build verification prompt
        context_text = "\n".join([
            f"[{i+1}] {ctx.get('text', '')[:500]}"
            for i, ctx in enumerate(contexts[:5])
        ])

        prompt = f"""Bạn là thẩm phán đánh giá tính chính xác.

Ngữ cảnh:
{context_text}

Câu trả lời:
{answer[:1000]}

Câu trả lời có được hỗ trợ hoàn toàn bởi ngữ cảnh không?
Trả lời: SUPPORTED hoặc NOT_SUPPORTED
"""

        try:
            response = self.primary.generate_text(prompt, temperature=0.0, max_tokens=50)

            if 'NOT_SUPPORTED' in response.upper():
                return 'NOT_SUPPORTED', 0.0
            elif 'SUPPORTED' in response.upper():
                return 'SUPPORTED', 1.0
            else:
                return 'UNKNOWN', 0.5
        except Exception as e:
            logger.warning(f"Faithfulness verification failed: {e}")
            return 'ERROR', 0.0


# Alias for backward compatibility
FallbackRAGPipeline = RAGChain
RAGPipeline = RAGChain
