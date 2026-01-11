"""
Text Splitters for Structural Chunking

Splits Vietnamese legal documents by structure (Chapter, Article, Clause).
"""

from typing import List, Dict, Optional
import re
import hashlib

from ccba_rag.core.models import Chunk, ChunkLevel
from ccba_rag.core.constants import (
    CHAPTER_PATTERN, ARTICLE_PATTERN, CLAUSE_PATTERN,
    SECTION_LEVEL_1_PATTERN, SECTION_LEVEL_2_PATTERN,
    MAX_CHUNK_CHARS, MIN_CHUNK_CHARS
)
from ccba_rag.core.settings import settings
from ccba_rag.utils.logging import get_logger

logger = get_logger(__name__)


class StructuralSplitter:
    """
    Structural text splitter for Vietnamese legal documents.
    
    Splits by:
    - Chapters (CHƯƠNG)
    - Articles (ĐIỀU) 
    - Clauses (numbered paragraphs)
    - Recursive character splitting for long segments
    """
    
    def __init__(
        self,
        max_chars: int = None,
        min_chars: int = None,
        overlap: int = 200
    ):
        """
        Initialize splitter.
        
        Args:
            max_chars: Maximum chunk size in characters
            min_chars: Minimum chunk size
            overlap: Overlap between chunks
        """
        self.max_chars = max_chars or settings.max_chunk_chars
        self.min_chars = min_chars or settings.min_chunk_chars
        self.overlap = overlap
    
    def split(
        self,
        text: str,
        document_id: str,
        document_name: str,
        pages: List[Dict] = None
    ) -> List[Chunk]:
        """
        Split text into chunks with page tracking.
        
        Args:
            text: Full document text
            document_id: Document identifier
            document_name: Document name for metadata
            pages: Optional list of page info from loader
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            logger.warning(f"Document empty: {document_name}")
            return []
        
        # Track original length for coverage calculation
        original_length = len(text)
        
        chunks = []
        current_chapter = None
        current_article = None
        
        # Split by newlines first
        lines = text.split('\n')
        current_block = []
        current_start_page = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for chapter
            chapter_match = CHAPTER_PATTERN.match(line)
            if chapter_match:
                # Flush current block
                if current_block:
                    chunks.extend(self._create_chunks_from_block(
                        "\n".join(current_block),
                        document_id, document_name,
                        current_chapter, current_article,
                        current_start_page
                    ))
                    current_block = []
                
                current_chapter = chapter_match.group(1)
                if chapter_match.group(2):
                    current_chapter += f": {chapter_match.group(2).strip()}"
                current_block.append(line)
                continue
            
            # Check for article
            article_match = ARTICLE_PATTERN.match(line)
            if article_match:
                # Flush current block
                if current_block:
                    chunks.extend(self._create_chunks_from_block(
                        "\n".join(current_block),
                        document_id, document_name,
                        current_chapter, current_article,
                        current_start_page
                    ))
                    current_block = []
                
                current_article = article_match.group(1)
                if article_match.group(2):
                    current_article += f": {article_match.group(2).strip()}"
                current_block.append(line)
                continue
            
            current_block.append(line)
        
        # Flush remaining
        if current_block:
            chunks.extend(self._create_chunks_from_block(
                "\n".join(current_block),
                document_id, document_name,
                current_chapter, current_article,
                current_start_page
            ))
        
        # Add IDs and truncate metadata
        for i, chunk in enumerate(chunks):
            chunk.id = self._generate_chunk_id(document_id, i)
            # Truncate metadata to fit schema
            chunk.chapter = (chunk.chapter or "")[:500] if chunk.chapter else None
            chunk.article = (chunk.article or "")[:500] if chunk.article else None
        
        # Log coverage
        chunk_chars = sum(len(c.text) for c in chunks)
        coverage = (chunk_chars / original_length * 100) if original_length > 0 else 0
        logger.info(f"Split {document_name} into {len(chunks)} chunks (coverage: {coverage:.1f}%)")
        return chunks
    
    def _create_chunks_from_block(
        self,
        text: str,
        document_id: str,
        document_name: str,
        chapter: Optional[str],
        article: Optional[str],
        page_number: int,
        existing_chunks: List[Chunk] = None
    ) -> List[Chunk]:
        """
        Create chunks from a text block, recursively splitting if needed.
        
        IMPORTANT: Never drops content. Small chunks are merged with previous.
        """
        chunks = existing_chunks or []
        
        if len(text) <= self.max_chars:
            # Single chunk - check if should merge with previous
            if len(text) < self.min_chars and chunks:
                # Merge with previous chunk instead of dropping
                chunks[-1].text += "\n" + text
                chunks[-1].token_count = len(chunks[-1].text.split())
            else:
                # Create new chunk (even if small, to prevent content loss)
                chunks.append(Chunk(
                    id="",  # Set later
                    document_id=document_id,
                    document_name=document_name,
                    text=text,
                    page_number=page_number,
                    chapter=chapter,
                    article=article,
                    level=ChunkLevel.ARTICLE if article else ChunkLevel.CHAPTER,
                    full_context=self._build_context_breadcrumb(chapter, article),
                    token_count=len(text.split())
                ))
        else:
            # Recursively split long text
            sub_texts = self._recursive_split(text)
            for i, sub_text in enumerate(sub_texts):
                if len(sub_text) < self.min_chars and chunks:
                    # Merge small sub-chunk with previous
                    chunks[-1].text += "\n" + sub_text
                    chunks[-1].token_count = len(chunks[-1].text.split())
                else:
                    chunks.append(Chunk(
                        id="",
                        document_id=document_id,
                        document_name=document_name,
                        text=sub_text,
                        page_number=page_number,
                        chapter=chapter,
                        article=article,
                        clause=str(i + 1) if len(sub_texts) > 1 else None,
                        level=ChunkLevel.CLAUSE,
                        full_context=self._build_context_breadcrumb(chapter, article),
                        token_count=len(sub_text.split())
                    ))
        
        return chunks
    
    def _recursive_split(self, text: str) -> List[str]:
        """Recursively split text to fit max_chars with overlap."""
        if len(text) <= self.max_chars:
            return [text]
        
        # Try splitting by paragraphs
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            chunks = []
            current = []
            current_len = 0
            
            for para in paragraphs:
                if current_len + len(para) > self.max_chars and current:
                    chunk_text = '\n\n'.join(current)
                    chunks.append(chunk_text)
                    
                    # Keep last paragraph for overlap
                    if self.overlap > 0 and current:
                        overlap_text = current[-1] if len(current[-1]) <= self.overlap else current[-1][-self.overlap:]
                        current = [overlap_text]
                        current_len = len(overlap_text)
                    else:
                        current = []
                        current_len = 0
                        
                current.append(para)
                current_len += len(para)
            
            if current:
                chunks.append('\n\n'.join(current))
            
            return chunks
        
        # Fall back to sentence splitting with overlap
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = []
        current_len = 0
        
        for sent in sentences:
            if current_len + len(sent) > self.max_chars and current:
                chunks.append(' '.join(current))
                
                # Keep last sentence for overlap
                if self.overlap > 0 and current:
                    current = [current[-1]]
                    current_len = len(current[-1])
                else:
                    current = []
                    current_len = 0
                    
            current.append(sent)
            current_len += len(sent)
        
        if current:
            chunks.append(' '.join(current))
        
        return chunks
    
    def _build_context_breadcrumb(
        self,
        chapter: Optional[str],
        article: Optional[str]
    ) -> str:
        """Build context breadcrumb for chunk."""
        parts = []
        if chapter:
            parts.append(f"Chương {chapter}")
        if article:
            parts.append(f"Điều {article}")
        return " > ".join(parts) if parts else ""
    
    def _generate_chunk_id(self, document_id: str, index: int) -> str:
        """Generate unique chunk ID."""
        return hashlib.md5(f"{document_id}_{index}".encode()).hexdigest()[:16]


# Alias for backward compatibility
RecursiveStructuralSplitter = StructuralSplitter
LayoutChunker = StructuralSplitter
