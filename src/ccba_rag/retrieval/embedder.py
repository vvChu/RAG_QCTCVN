"""
BGE-M3 Embedding Model with Hybrid Retrieval Support

Implements Dense + Sparse embeddings for semantic and lexical search.
CPU-optimized with configurable sequence length and batch size.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from ccba_rag.core.base import BaseEmbedder
from ccba_rag.core.settings import settings
from ccba_rag.utils.logging import get_logger

logger = get_logger(__name__)


class BGEEmbedder(BaseEmbedder):
    """
    BGE-M3 embedding model wrapper with hybrid (dense + sparse) support.

    Features:
    - Dense embeddings (1024 dim) for semantic search
    - Sparse lexical weights for keyword matching
    - CPU-optimized: sequence length 1024 tokens, batch size 64

    Configuration via settings:
    - bge_model_name: Model name (default: BAAI/bge-m3)
    - bge_max_length: Maximum sequence length (default: 1024)
    - bge_use_fp16: Use FP16 precision (default: False for CPU)
    - bge_batch_size: Default batch size (default: 64)
    """

    _instance: Optional['BGEEmbedder'] = None
    _model = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_fp16: Optional[bool] = None,
        max_length: Optional[int] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None
    ):
        """
        Initialize BGE-M3 model with CPU optimizations.

        Args:
            model_name: HuggingFace model name
            use_fp16: Use FP16 precision (False for CPU)
            max_length: Maximum sequence length (1024 for CPU)
            device: 'cuda', 'cpu', or None (auto-detect)
            batch_size: Default batch size for encoding
        """
        # Skip re-initialization if model already loaded
        if self._model is not None:
            return

        # Load from settings with overrides
        self.model_name = model_name or settings.bge_model_name
        self.max_length = max_length or settings.bge_max_length
        self.use_fp16 = use_fp16 if use_fp16 is not None else settings.bge_use_fp16
        self.batch_size_default = batch_size or settings.bge_batch_size
        self.device = device

        logger.info(f"Loading BGE-M3 model: {self.model_name}")
        logger.info(f"Config: max_length={self.max_length}, batch_size={self.batch_size_default}, fp16={self.use_fp16}")

        try:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
                device=device
            )
            logger.info("BGE-M3 model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {e}")
            raise

    @property
    def model(self):
        """Get the underlying model instance."""
        return self._model

    def encode_queries(
        self,
        queries: str | List[str],
        batch_size: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Encode queries into dense and sparse embeddings.

        Args:
            queries: Query string or list of queries
            batch_size: Batch size for encoding

        Returns:
            Dict with 'dense' (N, 1024) and 'sparse' (list of dicts) keys
        """
        if isinstance(queries, str):
            queries = [queries]

        batch_size = batch_size or self.batch_size_default

        output = self.model.encode(
            queries,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            max_length=self.max_length,
            batch_size=batch_size
        )

        return {
            'dense': output['dense_vecs'],
            'sparse': output['lexical_weights']
        }

    def encode_documents(
        self,
        documents: str | List[str],
        batch_size: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Encode documents into dense and sparse embeddings.

        Args:
            documents: Document string or list of documents
            batch_size: Batch size for encoding

        Returns:
            Dict with 'dense' and 'sparse' keys
        """
        if isinstance(documents, str):
            documents = [documents]

        batch_size = batch_size or self.batch_size_default

        output = self.model.encode(
            documents,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            max_length=self.max_length,
            batch_size=batch_size
        )

        return {
            'dense': output['dense_vecs'],
            'sparse': output['lexical_weights']
        }

    def get_embedding_dim(self) -> int:
        """Return the dimension of dense embeddings (1024 for BGE-M3)."""
        return 1024

    def encode_all(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        """
        Encode texts for bulk indexing with progress tracking.

        Args:
            texts: List of text strings
            batch_size: Batch size
            show_progress: Show progress bar

        Returns:
            Tuple of (dense_embeddings as list, sparse_weights as list of dicts)
        """
        batch_size = batch_size or self.batch_size_default

        all_dense = []
        all_sparse = []

        num_batches = (len(texts) + batch_size - 1) // batch_size

        iterator = range(num_batches)
        if show_progress and num_batches > 1:
            iterator = tqdm(
                iterator,
                desc=f"Encoding ({batch_size} x {self.max_length})"
            )

        for i in iterator:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]

            output = self.encode_documents(batch_texts, batch_size=len(batch_texts))

            all_dense.append(output['dense'])
            all_sparse.extend(output['sparse'])

        # Concatenate and convert to list format
        dense_embeddings = np.vstack(all_dense)
        dense_list = dense_embeddings.tolist()

        return dense_list, all_sparse

    # Alias for backward compatibility
    def encode_for_indexing(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Legacy alias for encode_all, returns numpy array."""
        dense_list, sparse_list = self.encode_all(texts, batch_size, show_progress)
        return np.array(dense_list), sparse_list


class HybridSearchScorer:
    """
    Combines scores from Dense and Sparse retrieval using Reciprocal Rank Fusion (RRF).

    RRF is robust and doesn't require score normalization.
    """

    def __init__(self, k: int = 60):
        """
        Args:
            k: Constant for score stabilization (typically 60).
        """
        self.k = k

    def fuse_results(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        id_key: str = 'id'
    ) -> List[Dict]:
        """
        Combine two ranked result lists using RRF.

        Args:
            dense_results: Results sorted by dense score
            sparse_results: Results sorted by sparse score
            id_key: Key to use for document ID

        Returns:
            Combined results sorted by RRF score
        """
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict] = {}

        # Process dense results
        for rank, doc in enumerate(dense_results):
            doc_id = doc[id_key]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
                doc_map[doc_id] = doc
            rrf_scores[doc_id] += 1.0 / (self.k + rank + 1)

        # Process sparse results
        for rank, doc in enumerate(sparse_results):
            doc_id = doc[id_key]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
                doc_map[doc_id] = doc
            rrf_scores[doc_id] += 1.0 / (self.k + rank + 1)

        # Build fused results
        fused_results = []
        for doc_id, score in rrf_scores.items():
            doc = doc_map[doc_id].copy()
            doc['rrf_score'] = score
            fused_results.append(doc)

        # Sort by RRF score descending
        fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)

        return fused_results
