"""
Query Service - Orchestrates RAG queries with caching

Provides a higher-level interface for query execution.
"""

from typing import List, Dict, Any, Optional
from functools import lru_cache
import hashlib
import time

from ccba_rag.utils.logging import get_logger

logger = get_logger(__name__)


class QueryService:
    """
    Service for executing RAG queries with optional caching.
    
    Features:
    - In-memory query cache
    - Retrieval-only mode
    - Full RAG mode
    - Statistics tracking
    """
    
    def __init__(
        self,
        chain,
        enable_cache: bool = True,
        cache_ttl: int = 300
    ):
        """
        Initialize query service.
        
        Args:
            chain: RAGChain instance
            enable_cache: Enable query caching
            cache_ttl: Cache TTL in seconds
        """
        self.chain = chain
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        
        self._cache: Dict[str, Any] = {}
        self._cache_times: Dict[str, float] = {}
    
    def query(
        self,
        question: str,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a RAG query.
        
        Args:
            question: User question
            use_cache: Use cached results if available
            **kwargs: Additional arguments for chain.query()
            
        Returns:
            Query result dict
        """
        # Check cache
        if use_cache and self.enable_cache:
            cache_key = self._cache_key(question, kwargs)
            cached = self._get_cached(cache_key)
            if cached:
                cached['from_cache'] = True
                return cached
        
        # Execute query
        result = self.chain.query(question, **kwargs)
        result['from_cache'] = False
        
        # Cache result
        if self.enable_cache and result.get('answer'):
            self._set_cached(cache_key, result)
        
        return result
    
    def query_retrieve_only(
        self,
        question: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute retrieval without generation.
        
        Args:
            question: Search query
            **kwargs: Retrieval parameters
            
        Returns:
            Dict with contexts and stats
        """
        contexts, stats = self.chain.retriever.retrieve(question, **kwargs)
        return {
            'query': question,
            'contexts': contexts,
            'stats': stats,
            'answer': None
        }
    
    def _cache_key(self, question: str, kwargs: dict) -> str:
        """Generate cache key from question and params."""
        key_str = f"{question}_{kwargs}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached result if valid."""
        if key not in self._cache:
            return None
        
        cached_time = self._cache_times.get(key, 0)
        if time.time() - cached_time > self.cache_ttl:
            del self._cache[key]
            del self._cache_times[key]
            return None
        
        return self._cache[key].copy()
    
    def _set_cached(self, key: str, value: Dict):
        """Cache a result."""
        self._cache[key] = value.copy()
        self._cache_times[key] = time.time()
    
    def clear_cache(self):
        """Clear all cached results."""
        self._cache.clear()
        self._cache_times.clear()
    
    def _display_results(self, result: Dict):
        """Display query results."""
        print("\n" + "=" * 60)
        print("ANSWER:")
        print(result.get('answer', 'No answer'))
        print("=" * 60)
