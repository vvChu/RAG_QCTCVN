from pathlib import Path
from typing import Any, List

from ccba_rag.ingestion.metadata_extractor import DocumentMetadataExtractor
from ccba_rag.ingestion.parsers import DocumentParser
from ccba_rag.ingestion.utils import generate_document_id
from ccba_rag.utils.logging import get_logger

from .base import PipelineContext, PipelineStage

logger = get_logger(__name__)

# --- Stage 1: Load ---
class LoadStage(PipelineStage):
    def __init__(self, parser: DocumentParser):
        self.parser = parser

    async def run(self, context: PipelineContext, file_path: str) -> Any:
        logger.info(f"Pipeline: Loading {Path(file_path).name}")
        try:
             doc_data = self.parser.parse(file_path)
             context.stats['pages'] = doc_data.get('stats', {}).get('total_pages', 0)
             context.stats['total_chars'] = doc_data.get('stats', {}).get('total_chars', 0)
             return doc_data
        except Exception as e:
            logger.error(f"LoadStage failed: {e}")
            raise

# --- Stage 2: Metadata ---
class MetadataStage(PipelineStage):
    def __init__(self):
        self.extractor = DocumentMetadataExtractor(use_llm_fallback=False)

    async def run(self, context: PipelineContext, doc_data: dict) -> Any:
        file_name = Path(context.file_path).name
        text = doc_data['text']

        # Extract metadata
        meta = self.extractor.extract(text, file_name)

        # Update context
        context.metadata.update({
            'document_code': meta.document_code,
            'document_type': meta.document_type,
            'amendment_number': meta.amendment_number,
            'amends_document': meta.amends_document
        })

        # Generator Doc ID
        doc_id = generate_document_id(context.file_path, meta.document_code)
        context.metadata['document_id'] = doc_id

        return doc_data

# --- Stage 3: Chunk ---
class ChunkStage(PipelineStage):
    def __init__(self, chunker):
        self.chunker = chunker

    async def run(self, context: PipelineContext, doc_data: dict) -> Any:
        doc_id = context.metadata.get('document_id')
        file_name = Path(context.file_path).name

        chunks = self.chunker.split(
            text=doc_data['text'],
            document_id=doc_id,
            document_name=file_name,
            pages=doc_data.get('pages', [])
        )

        if not chunks:
            logger.warning(f"No chunks created for {file_name}")
            return None

        # Attach metadata to chunks
        for chunk in chunks:
            if context.metadata.get('document_code'):
                chunk.document_code = context.metadata['document_code']

        context.stats['chunks_created'] = len(chunks)
        return chunks

# --- Stage 4: Embed ---
class EmbedStage(PipelineStage):
    def __init__(self, embedder):
        self.embedder = embedder

    async def run(self, context: PipelineContext, chunks: List[Any]) -> Any:
        texts = [chunk.text for chunk in chunks]

        logger.info(f"Pipeline: Embedding {len(chunks)} chunks... (Batch size: {self.embedder.batch_size_default})")

        # Run blocking embedding in executor to avoid freezing async loop
        import asyncio
        loop = asyncio.get_running_loop()

        # encode_all returns tuple (dense, sparse)
        dense, sparse = await loop.run_in_executor(
            None, # Default executor (ThreadPool)
            lambda: self.embedder.encode_all(texts, show_progress=False)
        )

        logger.info("Pipeline: Embedding complete.")

        for chunk, d, s in zip(chunks, dense, sparse):
            chunk.dense_vector = d
            chunk.sparse_vector = s

        return chunks

# --- Stage 5: Store ---
class VectorStoreStage(PipelineStage):
    def __init__(self, vector_db):
        self.vector_db = vector_db

    async def run(self, context: PipelineContext, chunks: List[Any]) -> Any:
        try:
            logger.info(f"Pipeline: Inserting {len(chunks)} chunks into Milvus...")
            self.vector_db.insert(chunks)
            logger.info(f"Pipeline: Inserted {len(chunks)} chunks successfully.")
            return chunks
        except Exception as e:
            logger.error(f"Store failed: {e}")
            raise

# --- Stage 6: Verify ---
class VerifyStage(PipelineStage):
    def __init__(self):
        pass

    async def run(self, context: PipelineContext, chunks: List[Any]) -> Any:
        stats = context.stats
        stats.get('total_chars', 0)
        # Re-calculate or get from upstream?
        # Ideally parsing stage sets total_chars.
        # But LoadStage does context.stats['pages']
        # We need total_chars in LoadStage

        # Let's fix LoadStage to return more stats if it doesn't already
        # Parser return 'stats' dict.

        chunk_chars = sum(len(c.text) for c in chunks)

        # We don't have total_chars in context.stats yet from LoadStage, only pages?
        # Let's check LoadStage implementation above.
        # It sets context.stats['pages'].
        # It should also set total_chars.

        coverage = 0
        if 'total_chars' in stats and stats['total_chars'] > 0:
             coverage = (chunk_chars / stats['total_chars'] * 100)

        context.stats['coverage'] = coverage
        context.stats['chunks_indexed'] = len(chunks)

        logger.info(f"Verification: {len(chunks)} chunks, Coverage: {coverage:.1f}%")

        if len(chunks) == 0:
            context.errors.append("Verification Failed: 0 chunks were indexed")

        return chunks
