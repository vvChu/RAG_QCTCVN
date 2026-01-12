"""Core module - Base classes, models, configuration, and system facade."""

from ccba_rag.core.base import BaseEmbedder, BaseGenerator, BaseReranker, BaseVectorDB
from ccba_rag.core.models import Chunk, Document, RAGResponse, RetrievalResult
from ccba_rag.core.settings import settings

__all__ = [
    "BaseVectorDB",
    "BaseEmbedder",
    "BaseReranker",
    "BaseGenerator",
    "Chunk",
    "Document",
    "RAGResponse",
    "RetrievalResult",
    "settings",
]
