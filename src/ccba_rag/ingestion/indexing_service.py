from datetime import datetime
from pathlib import Path
from typing import Optional

from ccba_rag.core.constants import SUPPORTED_EXTENSIONS
from ccba_rag.ingestion.parsers import DocumentParser
from ccba_rag.ingestion.pipeline import (
    ChunkStage,
    EmbedStage,
    IngestionPipeline,
    LoadStage,
    MetadataStage,
    VectorStoreStage,
    VerifyStage,
)
from ccba_rag.ingestion.report import IngestionReport
from ccba_rag.ingestion.splitters import StructuralSplitter
from ccba_rag.retrieval.embedder import BGEEmbedder
from ccba_rag.retrieval.vectorstores.milvus import MilvusStore
from ccba_rag.utils.logging import get_logger

logger = get_logger(__name__)

class IndexingService:
    """
    Modern indexing service using the Ingestion Pipeline.
    Supports AsyncIO and detailed reporting.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 200
    ):
        # Initialize components
        self.parser = DocumentParser()
        self.chunker = StructuralSplitter(max_chars=chunk_size, overlap=chunk_overlap)
        self.embedder = BGEEmbedder()
        self.vector_db = MilvusStore(collection_name=collection_name)

        # Build pipeline stages
        self.pipeline = IngestionPipeline([
            LoadStage(self.parser),
            MetadataStage(),
            ChunkStage(self.chunker),
            EmbedStage(self.embedder),
            VectorStoreStage(self.vector_db),
            VerifyStage()
        ])

    async def index_directory(
        self,
        directory: str,
        drop_existing: bool = False
    ) -> IngestionReport:
        """
        Index all documents in directory.
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise ValueError(f"Directory not found: {directory}")

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = IngestionReport(run_id=run_id)

        # Initialize DB
        self.vector_db.create_collection(drop_if_exists=drop_existing)

        # Find files
        files = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(dir_path.rglob(f"*{ext}"))
        files = sorted(files)

        logger.info(f"Starting indexing for {len(files)} files in {directory}")

        # Process sequentially for now (can be parallelized later with Semaphore)
        for file_path in files:
            context = await self.pipeline.run(str(file_path))
            report.add_result(context)

            # Save partial report
            report.save(f"ingestion_report_{run_id}.json")

        logger.info(f"Indexing complete. Success: {report.success_count}/{report.total_files}")
        return report
