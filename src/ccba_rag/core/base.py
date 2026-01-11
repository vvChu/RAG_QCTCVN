"""
Abstract Base Classes for RAG Components

This module defines interfaces that all component implementations must follow.
This enables:
- Easy component swapping (e.g., Milvus -> Qdrant, Gemini -> Groq)
- Better testing with mocks
- Clear contracts for component behavior
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
import numpy as np


class BaseVectorDB(ABC):
    """
    Abstract base class for vector database implementations.

    Implementations: MilvusStore, QdrantStore, ChromaStore
    """

    @abstractmethod
    def create_collection(self, dim: int = 1024, drop_if_exists: bool = False) -> None:
        """Create a collection/index for storing vectors."""
        pass

    @abstractmethod
    def create_index(self) -> None:
        """Create indexes on the collection for efficient search."""
        pass

    @abstractmethod
    def load_collection(self) -> None:
        """Load collection into memory for searching."""
        pass

    @abstractmethod
    def insert(self, chunks: List[Any]) -> List[str]:
        """
        Insert chunks with vectors and metadata.

        Args:
            chunks: List of Chunk objects with dense_vector and sparse_vector populated

        Returns:
            List of inserted document IDs
        """
        pass

    @abstractmethod
    def search(
        self,
        query_dense: List[float],
        query_sparse: Dict[int, float],
        top_k: int = 100,
        expr: Optional[str] = None
    ) -> List[Any]:
        """
        Perform hybrid search (dense + sparse with RRF fusion).

        Args:
            query_dense: Dense vector embedding
            query_sparse: Sparse vector (token_id -> weight)
            top_k: Number of results to return
            expr: Optional filter expression

        Returns:
            List of RetrievalResult objects
        """
        pass

    @abstractmethod
    def has_collection(self) -> bool:
        """Check if collection exists."""
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the collection."""
        pass

    @abstractmethod
    def delete_by_document(self, document_name: str) -> int:
        """Delete all chunks for a document. Returns count deleted."""
        pass


class BaseEmbedder(ABC):
    """
    Abstract base class for embedding models.

    Implementations: BGEEmbedder, GeminiEmbedder
    """

    @abstractmethod
    def encode_queries(
        self,
        queries: List[str],
        batch_size: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Encode queries into dense and sparse vectors.

        Returns:
            Dict with 'dense' (N, D) and 'sparse' (list of dicts) keys
        """
        pass

    @abstractmethod
    def encode_documents(
        self,
        documents: List[str],
        batch_size: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Encode documents into dense and sparse vectors.

        Returns:
            Dict with 'dense' and 'sparse' keys
        """
        pass

    @abstractmethod
    def encode_all(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        """
        Encode texts for bulk indexing.

        Returns:
            Tuple of (dense_embeddings, sparse_weights)
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the dimension of dense embeddings."""
        pass


class BaseReranker(ABC):
    """
    Abstract base class for reranking models.

    Implementations: BGEM3Reranker, CrossEncoderReranker
    """

    @abstractmethod
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
            documents: List of candidate documents (must have 'text' or 'rerank_text' key)
            top_n: Number of top documents to return

        Returns:
            Top N reranked documents with 'rerank_score' added
        """
        pass

    @abstractmethod
    def rerank_with_details(
        self,
        query: str,
        documents: List[Dict],
        top_n: int = 5
    ) -> Tuple[List[Dict], Dict]:
        """
        Rerank with detailed statistics.

        Returns:
            Tuple of (ranked_documents, stats_dict)
        """
        pass


class BaseGenerator(ABC):
    """
    Abstract base class for answer generation models.

    Implementations: GeminiGenerator, GroqGenerator, DeepSeekGenerator
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier."""
        pass

    @abstractmethod
    def generate(
        self,
        query: str,
        contexts: List[Dict],
        history: Optional[List[Dict]] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Generate answer based on query and contexts.

        Args:
            query: User question
            contexts: Retrieved context documents
            history: Optional conversation history
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dict containing:
            - 'answer': Generated text
            - 'model': Model identifier
            - 'prompt': Full prompt used
            - 'error': Error info if any (optional)
        """
        pass

    @abstractmethod
    def format_contexts(self, contexts: List[Dict]) -> str:
        """
        Format contexts into a string suitable for the prompt.

        Args:
            contexts: List of context documents

        Returns:
            Formatted context string
        """
        pass

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 1024
    ) -> str:
        """
        Direct text generation without RAG template.
        Default implementation - subclasses should override.
        """
        raise NotImplementedError("Subclass must implement generate_text")


class BaseChunker(ABC):
    """
    Abstract base class for document chunking.

    Implementations: StructuralSplitter, RecursiveCharacterSplitter
    """

    @abstractmethod
    def chunk_document(
        self,
        file_path: str,
        document_id: str,
        document_name: str
    ) -> List[Any]:
        """
        Process a document and return chunks.

        Args:
            file_path: Path to the document file
            document_id: Unique identifier for the document
            document_name: Human-readable document name

        Returns:
            List of Chunk objects
        """
        pass
