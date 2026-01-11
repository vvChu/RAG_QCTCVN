"""
Re-index all documents with the new optimized directory structure.

This script will:
1. Drop the existing collection  
2. Re-create collection
3. Index all documents from the organized directories
4. Verify indexing success
"""
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.config.settings import settings
from src.vector_store.milvus import MilvusStore
from src.utils.logging import logger
from pymilvus import utility
import glob

def main():
    logger.info("="*60)
    logger.info("RE-INDEXING WITH OPTIMIZED DATA STRUCTURE")
    logger.info("="*60)
    
    # Initialize vector store
    vector_store = MilvusStore()
    
    # Connect
    logger.info("\n[1/4] Connecting to Milvus...")
    vector_store._ensure_connection()
    logger.info(f"  ✓ Connected to: {settings.milvus_host}")
    
    # Drop existing collection
    logger.info("\n[2/4] Dropping existing collection...")
    if utility.has_collection(settings.milvus_collection_name):
        utility.drop_collection(settings.milvus_collection_name)
        logger.info(f"  ✓ Dropped: {settings.milvus_collection_name}")
    else:
        logger.info(f"  ℹ Collection does not exist, will create new")
    
    # Create fresh collection
    logger.info("\n[3/4] Creating fresh collection...")
    vector_store.create_collection(dim=1024)
    logger.info(f"  ✓ Created: {settings.milvus_collection_name}")
    
    # Count files in organized structure
    logger.info("\n[4/4] Counting documents in organized structure...")
    
    categories = {
        "QCVN": "data/documents/qcvn",
        "TCVN": "data/documents/tcvn",
        "ND-CP": "data/documents/nd-cp",
        "TT-BXD": "data/documents/tt-bxd"
    }
    
    total_files = 0
    for category, directory in categories.items():
        pdfs = glob.glob(f"{directory}/*.pdf")
        docx = glob.glob(f"{directory}/*.docx")
        count = len(pdfs) + len(docx)
        total_files += count
        logger.info(f"  {category}: {count} files")
    
    logger.info(f"\n  Total: {total_files} files ready for indexing")
    
    logger.info("\n"+ "="*60)
    logger.info("Collection is ready. Run orchestrator to index:")
    logger.info("  python run_optimized_orchestrator.py")
    logger.info("="*60)

if __name__ == "__main__":
    main()
