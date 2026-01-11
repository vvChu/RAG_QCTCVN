import asyncio
import sys
from pathlib import Path
sys.path.insert(0, 'src')

from ccba_rag.ingestion.pipeline import (
    IngestionPipeline, 
    LoadStage, 
    MetadataStage, 
    ChunkStage, 
    EmbedStage, 
    VectorStoreStage,
    VerifyStage
)
from ccba_rag.ingestion.parsers import DocumentParser
from ccba_rag.ingestion.splitters import StructuralSplitter
from ccba_rag.retrieval.embedder import BGEEmbedder
from ccba_rag.retrieval.vectorstores.milvus import MilvusStore
from ccba_rag.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

async def run_test():
    file_path = "data/documents/01_QUY_CHUAN/QCVN_01_2021_KIENTRUC_Quy_Hoach_Xay_Dung.pdf"
    
    # 1. Initialize Components
    logger.info("Initializing components...")
    parser = DocumentParser() # Default chain
    chunker = StructuralSplitter(max_chars=1024, overlap=200)
    embedder = BGEEmbedder() # Slow loading
    vector_db = MilvusStore(collection_name="test_pipeline_collection")
    
    # Create temp collection
    vector_db.create_collection(drop_if_exists=True)
    
    # 2. Build Pipeline
    logger.info("Building pipeline...")
    stages = [
        LoadStage(parser),
        MetadataStage(),
        ChunkStage(chunker),
        EmbedStage(embedder),
        VectorStoreStage(vector_db),
        VerifyStage()
    ]
    
    pipeline = IngestionPipeline(stages)
    
    # 3. Run
    logger.info(f"Running pipeline on {file_path}")
    result_context = await pipeline.run(file_path)
    
    # 4. Report
    print("\n" + "="*60)
    print("PIPELINE REPORT")
    print("="*60)
    print(f"File: {result_context.file_path}")
    print(f"Stats: {result_context.stats}")
    print(f"Metadata: {result_context.metadata.keys()}")
    print("-" * 30)
    
    if result_context.errors:
        print("❌ Errors:")
        for e in result_context.errors:
            print(f"  - {e}")
    else:
        print("✅ Success! No errors.")
        print(f"Chunks Indexed: {result_context.stats.get('chunks_indexed')}")
        print(f"Coverage: {result_context.stats.get('coverage'):.2f}%")

if __name__ == "__main__":
    asyncio.run(run_test())
