"""
Pydantic Domain Models for the CCBA RAG System

These models define the core data structures used throughout the pipeline.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TXT = "txt"


class ChunkLevel(str, Enum):
    """Hierarchical level of a chunk."""
    DOCUMENT = "document"
    CHAPTER = "chapter"
    ARTICLE = "article"
    SECTION = "section"
    CLAUSE = "clause"


class Document(BaseModel):
    """Represents a source document before processing."""
    id: str
    name: str
    file_path: str
    type: DocumentType
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        frozen = True


class Chunk(BaseModel):
    """
    Represents a processed text chunk ready for indexing.
    
    This is the primary data structure flowing through the pipeline.
    """
    id: str
    document_id: str
    document_name: str
    text: str
    page_number: int = 1
    
    # Structural metadata (Vietnamese legal documents)
    chapter: Optional[str] = None
    article: Optional[str] = None
    clause: Optional[str] = None
    
    # Document relationship metadata (extracted from content)
    document_code: Optional[str] = None  # e.g., "QCVN 06:2022/BXD"
    document_type: Optional[str] = None  # "original", "amendment"
    amendment_number: Optional[str] = None  # e.g., "1:2023"
    amends_document: Optional[str] = None  # Document being amended
    
    # Hierarchical fields
    parent_id: Optional[str] = None
    level: ChunkLevel = ChunkLevel.CLAUSE
    full_context: Optional[str] = None  # Breadcrumb: "Chương I > Điều 5 > Khoản 1"
    
    # Layout fields (for PDF rendering)
    bbox: Optional[List[float]] = None  # [x0, y0, x1, y1]
    
    # Token stats
    token_count: int = 0
    
    # Vectors (populated by embedder)
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict[int, float]] = None

    def get_citation(self) -> str:
        """Returns a formatted citation string for display."""
        parts = [self.document_name]
        if self.chapter:
            parts.append(f"Chương {self.chapter}")
        if self.article:
            parts.append(f"Điều {self.article}")
        if self.clause:
            parts.append(f"Khoản {self.clause}")
        return f"[{' - '.join(parts)}]"
    
    def to_context_dict(self) -> Dict[str, Any]:
        """Convert to dict format expected by generators."""
        return {
            "text": self.text,
            "document_name": self.document_name,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "chapter": self.chapter,
            "article": self.article,
            "clause": self.clause,
            "full_context": self.full_context,
            "bbox": self.bbox,
        }


class RetrievalResult(BaseModel):
    """Represents a single retrieved item from vector search."""
    chunk: Chunk
    score: float
    rank: int = 0
    
    class Config:
        arbitrary_types_allowed = True


class GenerationStats(BaseModel):
    """Statistics for a single RAG query execution."""
    encoding_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    
    candidates_count: int = 0
    final_count: int = 0
    
    primary_model: Optional[str] = None
    fallback_model: Optional[str] = None
    used_fallback: bool = False


class RAGResponse(BaseModel):
    """Final response from the RAG system."""
    answer: str
    citations: List[str] = Field(default_factory=list)
    contexts: List[Dict[str, Any]] = Field(default_factory=list)
    stats: GenerationStats = Field(default_factory=GenerationStats)
    prompt: Optional[str] = None
    
    # For verification
    faithfulness_score: Optional[float] = None
    faithfulness_verdict: Optional[str] = None


class QueryRequest(BaseModel):
    """API request model for queries."""
    query: str
    top_k: int = 100
    top_n: int = 5
    use_expansion: bool = False
    use_reranker: bool = False
    history: List[Dict[str, Any]] = Field(default_factory=list)
    filters: Optional[Dict[str, Any]] = None


class IndexRequest(BaseModel):
    """API request model for indexing."""
    directory: str = "data"
    drop_existing: bool = False


class FeedbackRequest(BaseModel):
    """API request model for user feedback."""
    query: str
    answer: str
    rating: str  # "positive" or "negative"
    model: str
    comment: Optional[str] = None
