"""
Base class for vector store implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class VectorStore(ABC):
    """
    Abstract base class for vector store implementations.

    This is a simplified interface - see ccba_rag.core.base.BaseVectorDB
    for the full interface specification.
    """

    @abstractmethod
    def has_collection(self) -> bool:
        """Check if collection exists."""
        pass

    @abstractmethod
    def create_collection(self, dim: int = 1024, drop_if_exists: bool = False) -> None:
        """Create the collection with schema."""
        pass

    @abstractmethod
    def load_collection(self) -> None:
        """Load collection into memory."""
        pass

    @abstractmethod
    def insert(self, chunks: List[Any]) -> List[str]:
        """Insert chunks and return IDs."""
        pass

    @abstractmethod
    def search(
        self,
        query_dense: List[float],
        query_sparse: Dict[int, float],
        top_k: int = 100,
        expr: Optional[str] = None
    ) -> List[Any]:
        """Perform hybrid search."""
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the collection."""
        pass

    @abstractmethod
    def delete_by_document(self, document_name: str) -> int:
        """Delete chunks by document name."""
        pass
