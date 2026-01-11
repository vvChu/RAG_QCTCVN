"""
Test updated ingestion pipeline with all improvements.
"""

import sys
sys.path.insert(0, 'src')

from pathlib import Path
from ccba_rag.ingestion.loaders import PDFLoader
from ccba_rag.ingestion.splitters import StructuralSplitter


def test_loader():
    """Test PDF loader with page tracking."""
    print("=" * 60)
    print("Testing PDF Loader with Page Tracking")
    print("=" * 60)
    
    # Find a PDF to test
    test_files = list(Path("data/documents").rglob("*.pdf"))[:3]
    
    if not test_files:
        print("No PDF files found")
        return
    
    loader = PDFLoader(use_llamaparse_for_scanned=False)  # Skip API for speed
    
    for pdf_path in test_files:
        print(f"\n--- {pdf_path.name} ---")
        result = loader.load(str(pdf_path))
        
        stats = result.get('stats', {})
        print(f"  Pages: {stats.get('total_pages', 0)}")
        print(f"  Chars: {stats.get('total_chars', 0)}")
        print(f"  Images: {stats.get('total_images', 0)}")
        print(f"  Tables: {stats.get('total_tables', 0)}")
        print(f"  Pages with text: {stats.get('pages_with_text', 0)}")
        print(f"  Empty pages: {stats.get('pages_empty', 0)}")
        print(f"  Is scanned: {result.get('metadata', {}).get('is_scanned', False)}")
        
        # Test chunking with pages
        if result.get('text'):
            splitter = StructuralSplitter()
            chunks = splitter.split(
                text=result['text'][:5000],  # First 5000 chars for speed
                document_id="test_id",
                document_name=pdf_path.name,
                pages=result.get('pages', [])
            )
            print(f"  Chunks created: {len(chunks)}")


def main():
    test_loader()
    print("\n" + "=" * 60)
    print("Pipeline test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
