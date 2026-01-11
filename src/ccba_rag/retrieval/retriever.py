"""
Hybrid Retriever with Two-Stage Pipeline

Stage 1: Hybrid Search (Dense + Sparse with RRF)
Stage 2: Reranking (optional, for higher accuracy)
"""

from typing import Optional, Any
import time

from ccba_rag.core.settings import settings
from ccba_rag.utils.logging import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """
    Two-Stage Retrieval Pipeline: Hybrid Retrieval -> Reranking

    Stage 1: Vector DB hybrid search (dense + sparse with RRF)
    Stage 2: Cross-encoder reranking (optional, configurable)

    Configuration:
    - settings.enable_reranker: Enable Stage 2 reranking
    - settings.reranker_top_k: Number of candidates from Stage 1
    - settings.reranker_top_n: Number of final results after Stage 2
    """

    def __init__(
        self,
        vector_db,
        embedder,
        reranker=None
    ):
        """
        Initialize the retriever.

        Args:
            vector_db: MilvusStore instance
            embedder: BGEEmbedder instance
            reranker: Optional reranker instance (BGEM3Reranker)
        """
        self.vector_db = vector_db
        self.embedder = embedder
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 100,
        top_n: int = 5,
        use_reranker: Optional[bool] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Execute the two-stage retrieval pipeline.

        Args:
            query: Search query
            filters: Optional metadata filters (e.g., {"document_name": ["QCVN 06"]})
            top_k: Number of candidates from Stage 1
            top_n: Number of final results
            use_reranker: Override settings.enable_reranker

        Returns:
            Tuple of (results, stats)
            - results: List of dicts with 'text', 'document_name', 'rerank_score', etc.
            - stats: Pipeline timing statistics
        """
        pipeline_stats = {}

        # Stage 1: Encoding
        start_time = time.time()
        dense_list, sparse_list = self.embedder.encode_all([query], show_progress=False)
        query_dense = dense_list[0]
        query_sparse = sparse_list[0]
        encoding_time = (time.time() - start_time) * 1000
        pipeline_stats['encoding_ms'] = encoding_time

        # Build filter expression
        expr = self._build_filter_expr(filters)

        # Stage 2: Vector Search (with RRF fusion in MilvusStore)
        start_time = time.time()
        retrieval_results = self.vector_db.search(
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=top_k,
            expr=expr
        )
        retrieval_time = (time.time() - start_time) * 1000
        pipeline_stats['retrieval_ms'] = retrieval_time
        pipeline_stats['candidates_count'] = len(retrieval_results)

        # Convert RetrievalResult to Dict format
        candidates = []
        for res in retrieval_results:
            chunk = res.chunk
            candidates.append({
                'text': chunk.text,
                'document_name': chunk.document_name,
                'document_id': chunk.document_id,
                'page_number': chunk.page_number,
                'chapter': chunk.chapter,
                'article': chunk.article,
                'clause': chunk.clause,
                'full_context': chunk.full_context,
                'retrieval_score': res.score,
                'bbox': chunk.bbox,
                # Truncated text for reranking efficiency
                'rerank_text': self._prepare_rerank_text(chunk)
            })

        # Stage 3: Reranking (optional)
        should_rerank = use_reranker if use_reranker is not None else settings.enable_reranker

        if not should_rerank or not self.reranker:
            # Skip reranking - just take top_n from retrieval
            final_results = candidates[:top_n]
            for doc in final_results:
                doc['rerank_score'] = doc['retrieval_score']
            pipeline_stats['reranking_ms'] = 0
        else:
            # Apply reranking
            final_results, rerank_stats = self.reranker.rerank_with_details(
                query=query,
                documents=candidates,
                top_n=top_n
            )
            pipeline_stats['reranking_ms'] = rerank_stats.get('latency_ms', 0)

        pipeline_stats['final_count'] = len(final_results)
        pipeline_stats['total_ms'] = (
            pipeline_stats['encoding_ms'] +
            pipeline_stats['retrieval_ms'] +
            pipeline_stats['reranking_ms']
        )

        return final_results, pipeline_stats

    def _build_filter_expr(self, filters: Optional[Dict[str, Any]]) -> Optional[str]:
        """Build Milvus filter expression from filter dict."""
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                values_str = ",".join([f'"{v}"' for v in value])
                conditions.append(f'{key} in [{values_str}]')
            elif isinstance(value, str):
                conditions.append(f'{key} == "{value}"')

        return " and ".join(conditions) if conditions else None

    def _prepare_rerank_text(self, chunk, max_chars: int = 2000) -> str:
        """Prepare text for reranking with context."""
        if chunk.full_context:
            text = f"{chunk.full_context}\n{chunk.text}"
        else:
            text = chunk.text
        return text[:max_chars]


# Alias for backward compatibility
TwoStageRetriever = HybridRetriever
