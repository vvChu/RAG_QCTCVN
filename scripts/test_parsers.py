import sys
from pathlib import Path
from typing import List
# Add src to python path
sys.path.insert(0, 'src')

from ccba_rag.ingestion.parsers import DocumentParser, PyMuPDFStrategy, LlamaParseStrategy
from ccba_rag.utils.logging import configure_logging

configure_logging()

def test_parser():
    # Test file - pick one that exists
    test_pdf = Path("data/documents/01_QUY_CHUAN/QCVN_01_2021_KIENTRUC_Quy_Hoach_Xay_Dung.pdf")
    
    if not test_pdf.exists():
        print(f"Test file not found: {test_pdf}")
        return

    print(f"Testing parsers on: {test_pdf.name}")
    print("="*60)

    # 1. Test PyMuPDF explicitly
    print("\n1. Testing PyMuPDFStrategy...")
    try:
        strategy = PyMuPDFStrategy()
        result = strategy.parse(test_pdf)
        print(f"✅ Success! Pages: {len(result['pages'])}")
        print(f"   Chars: {result['stats']['total_chars']}")
        print(f"   Images: {result['stats']['total_images']}")
    except Exception as e:
        print(f"❌ Failed: {e}")

    # 2. Test Default Chain (which starts with LlamaParse)
    # WARNING: This consumes credits/time.
    print("\n2. Testing DocumentParser (Default Chain - LlamaParse First)...")
    try:
        parser = DocumentParser() # Uses LlamaParse -> PyMuPDF -> Gemini
        result = parser.parse(str(test_pdf))
        print(f"✅ Success! Parser used: {result['stats'].get('parser')}")
        print(f"   Pages: {len(result['pages'])}")
        print(f"   Chars: {result['stats']['total_chars']}")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    test_parser()
