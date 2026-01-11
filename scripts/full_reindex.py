import asyncio
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, 'src')

from ccba_rag.ingestion.indexing_service import IndexingService
from ccba_rag.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

async def run_full_reindex():
    data_dir = "data/documents"
    collection_name = "legal_docs_v2" # Using a new collection name for safety/versioning
    
    logger.info("="*60)
    logger.info("STARTING FULL RE-INDEXING (V2 Pipeline)")
    logger.info("="*60)
    logger.info(f"Source Directory: {data_dir}")
    logger.info(f"Target Collection: {collection_name}")
    
    # Initialize Service
    # Note: Explicitly setting smaller batch size in BGEEmbedder via settings would be ideal, 
    # but IndexingService init doesn't pass it through yet.
    # We will patch settings temporarily or rely on default.
    from ccba_rag.core.settings import settings
    settings.bge_batch_size = 4 # Reduce batch size for safety during re-index
    
    service = IndexingService(
        collection_name=collection_name,
        chunk_size=1024,
        chunk_overlap=200
    )
    
    # Run Indexing
    try:
        # distinct from sync_directory, index_directory re-processes everything
        report = await service.index_directory(
            directory=data_dir,
            drop_existing=True # Clean start
        )
        
        # Save Final Report
        report_path = "full_reindex_report.json"
        report.save(report_path)
        
        print("\n" + "="*60)
        print("RE-INDEXING COMPLETE")
        print("="*60)
        print(f"Total Files: {report.total_files}")
        print(f"Success:     {report.success_count}")
        print(f"Failed:      {report.failure_count}")
        print(f"Total Chunks:{report.total_chunks}")
        print(f"Report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Re-indexing failed: {e}")
        raise

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_full_reindex())
