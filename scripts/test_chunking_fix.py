"""
Test chunking coverage after fix.
"""

import sys
sys.path.insert(0, 'src')

from ccba_rag.ingestion.splitters import StructuralSplitter

# Test with various document types
TEST_DOCS = [
    # Short document (was rejected before)
    {
        "name": "short_doc.pdf",
        "text": "Điều 1. Định nghĩa ngắn về PCCC.",
        "expected_min_chunks": 1
    },
    # Normal document with articles
    {
        "name": "normal_doc.pdf",
        "text": """
CHƯƠNG I: QUY ĐỊNH CHUNG
Điều 1. Phạm vi điều chỉnh
Quy chuẩn này quy định về an toàn cháy cho nhà và công trình.
Điều 2. Đối tượng áp dụng
Quy chuẩn này áp dụng cho tất cả các tổ chức, cá nhân liên quan đến thiết kế, xây dựng, sử dụng nhà và công trình.
CHƯƠNG II: QUY ĐỊNH KỸ THUẬT
Điều 3. Yêu cầu chung
1. Nhà và công trình phải được thiết kế, xây dựng đảm bảo an toàn cháy.
2. Các giải pháp chống cháy phải phù hợp với loại hình công trình.
Điều 4. Phân loại công trình
Công trình được phân thành các nhóm sau: nhà ở, công trình công cộng, công trình công nghiệp.
        """,
        "expected_min_chunks": 3
    },
    # Document with small sections (was dropped before)
    {
        "name": "mixed_doc.pdf",
        "text": """
Điều 1. Phạm vi
Quy định này áp dụng cho tất cả công trình xây dựng.
Điều 2. Giải thích từ ngữ
a) PCCC: Phòng cháy chữa cháy
b) BXD: Bộ Xây dựng
Điều 3. Nguyên tắc
Đảm bảo an toàn.
        """,
        "expected_min_chunks": 1
    },
]


def test_chunking():
    print("=" * 60)
    print("Testing Fixed Chunking Strategy")
    print("=" * 60)
    
    splitter = StructuralSplitter(max_chars=2000, min_chars=100)
    
    for doc in TEST_DOCS:
        print(f"\n--- {doc['name']} ---")
        print(f"Input length: {len(doc['text'])} chars")
        
        chunks = splitter.split(
            text=doc['text'],
            document_id="test_id",
            document_name=doc['name']
        )
        
        print(f"Chunks created: {len(chunks)}")
        
        # Calculate coverage
        chunk_chars = sum(len(c.text) for c in chunks)
        original_chars = len(doc['text'].strip())
        coverage = (chunk_chars / original_chars * 100) if original_chars > 0 else 0
        
        print(f"Coverage: {coverage:.1f}%")
        
        if len(chunks) >= doc['expected_min_chunks']:
            print(f"✓ PASS - at least {doc['expected_min_chunks']} chunks")
        else:
            print(f"✗ FAIL - expected at least {doc['expected_min_chunks']} chunks")
        
        # Show first chunk preview
        if chunks:
            preview = chunks[0].text[:100] + "..." if len(chunks[0].text) > 100 else chunks[0].text
            print(f"First chunk: {preview}")
    
    print("\n" + "=" * 60)
    print("Chunking test complete!")


if __name__ == "__main__":
    test_chunking()
