"""
Reranker Implementations for Cross-Encoder Reranking

Supports multiple reranker backends:
- BGEM3Reranker: Fast lightweight reranker using BGE-M3 embeddings
- CrossEncoderReranker: Full cross-encoder (slower, more accurate)
"""

from typing import Optional
import time

from ccba_rag.core.base import BaseReranker
from ccba_rag.core.settings import settings
from ccba_rag.utils.logging import get_logger

logger = get_logger(__name__)


class BGEM3Reranker(BaseReranker):
    """
    Lightweight reranker using FlagReranker from FlagEmbedding.

    Faster than full cross-encoders, suitable for CPU inference.
    """

    _instance: Optional['BGEM3Reranker'] = None
    _model = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_fp16: bool = False,
        batch_size: Optional[int] = None
    ):
        """
        Initialize BGE-M3 Reranker.

        Args:
            model_name: Reranker model name
            use_fp16: Use FP16 precision
            batch_size: Batch size for reranking
        """
        if self._model is not None:
            return

        self.model_name = model_name or settings.reranker_model_name
        self.use_fp16 = use_fp16
        self.batch_size = batch_size or settings.reranker_batch_size

        logger.info(f"Loading BGEM3Reranker: {self.model_name}")

        try:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(
                self.model_name,
                use_fp16=self.use_fp16
            )
            logger.info("BGEM3Reranker loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            raise

    @property
    def model(self):
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_n: int = 5
    ) -> List[Dict]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            documents: List of documents (must have 'text' or 'rerank_text')
            top_n: Number of top results to return
            
        Returns:
            Top N reranked documents with 'rerank_score' added
        """
        if not documents:
            return []
        
        # Prepare query-document pairs
        pairs = [
            [query, doc.get('rerank_text', doc.get('text', ''))]
            for doc in documents
        ]
        
        # Compute scores
        scores = self.model.compute_score(pairs, batch_size=self.batch_size)
        
        # Handle single result case
        if isinstance(scores, (int, float)):
            scores = [scores]
        
        # Attach scores to documents
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # Sort by score descending
        ranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        return ranked[:top_n]
    
    def rerank_with_details(
        self,
        query: str,
        documents: List[Dict],
        top_n: int = 5
    ) -> Tuple[List[Dict], Dict]:
        """
        Rerank with timing and statistics.
        
        Returns:
            Tuple of (ranked_documents, stats)
        """
        start_time = time.time()
        ranked_docs = self.rerank(query, documents, top_n)
        latency_ms = (time.time() - start_time) * 1000
        
        stats = {
            'input_count': len(documents),
            'output_count': len(ranked_docs),
            'latency_ms': latency_ms,
            'avg_rerank_score': sum(d['rerank_score'] for d in ranked_docs) / len(ranked_docs) if ranked_docs else 0,
            'max_rerank_score': max(d['rerank_score'] for d in ranked_docs) if ranked_docs else 0,
            'min_rerank_score': min(d['rerank_score'] for d in ranked_docs) if ranked_docs else 0,
        }
        
        return ranked_docs, stats


class CrossEncoderReranker(BaseReranker):
    """
    Full cross-encoder reranker using Transformers.
    
    More accurate but slower than BGEM3Reranker.
    Suitable for GPU inference or when accuracy is critical.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        batch_size: int = 64,
        max_length: int = 256
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name
            device: 'cuda', 'cpu', or None (auto-detect)
            batch_size: Batch size for inference
            max_length: Maximum sequence length
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        logger.info(f"Loading CrossEncoderReranker: {model_name} on {device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("CrossEncoderReranker loaded successfully!")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_n: int = 5
    ) -> List[Dict]:
        """Rerank using cross-encoder."""
        import torch
        
        if not documents:
            return []
        
        pairs = [
            (query, doc.get('rerank_text', doc.get('text', '')))
            for doc in documents
        ]
        
        scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i:i + self.batch_size]
                
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                batch_scores = torch.sigmoid(outputs.logits[:, 0]).cpu().numpy()
                scores.extend(batch_scores.tolist())
        
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = score
        
        ranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        return ranked[:top_n]
    
    def rerank_with_details(
        self,
        query: str,
        documents: List[Dict],
        top_n: int = 5
    ) -> Tuple[List[Dict], Dict]:
        """Rerank with timing and statistics."""
        start_time = time.time()
        ranked_docs = self.rerank(query, documents, top_n)
        latency_ms = (time.time() - start_time) * 1000
        
        stats = {
            'input_count': len(documents),
            'output_count': len(ranked_docs),
            'latency_ms': latency_ms,
            'avg_rerank_score': sum(d['rerank_score'] for d in ranked_docs) / len(ranked_docs) if ranked_docs else 0,
            'max_rerank_score': max(d['rerank_score'] for d in ranked_docs) if ranked_docs else 0,
            'min_rerank_score': min(d['rerank_score'] for d in ranked_docs) if ranked_docs else 0,
        }
        
        return ranked_docs, stats


# Aliases for backward compatibility
VietnameseReranker = CrossEncoderReranker
