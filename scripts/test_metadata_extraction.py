"""
Test Document Metadata Extraction

Verifies extraction of QCVN/TCVN codes and amendment relationships.
"""

import sys
sys.path.insert(0, 'src')

from ccba_rag.ingestion.metadata_extractor import DocumentMetadataExtractor, extract_document_metadata

# Test cases
TEST_CASES = [
    # Original document
    {
        "text": """
QCVN 06:2022/BXD
QUY CHUẨN KỸ THUẬT QUỐC GIA VỀ AN TOÀN CHÁY CHO NHÀ VÀ CÔNG TRÌNH
Ban hành kèm theo Thông tư số 06/2022/TT-BXD ngày 05/10/2022
Có hiệu lực từ ngày 01/01/2023
        """,
        "expected": {
            "document_code": "QCVN 06:2022/BXD",
            "document_type": "original",
        }
    },
    # Amendment document
    {
        "text": """
SỬA ĐỔI 1:2023 QCVN 06:2022/BXD
QUY CHUẨN KỸ THUẬT QUỐC GIA VỀ AN TOÀN CHÁY CHO NHÀ VÀ CÔNG TRÌNH
Sửa đổi 1:2023 của QCVN 06:2022/BXD
Ban hành kèm theo Thông tư số XX/2023/TT-BXD
        """,
        "expected": {
            "document_code": "QCVN 06:2022/BXD",
            "document_type": "amendment",
            "amendment_number": "1:2023",
            "amends_document": "QCVN 06:2022/BXD"
        }
    },
    # TCVN standard
    {
        "text": """
TCVN 4451:2012
TIÊU CHUẨN VIỆT NAM
NHÀ Ở - NGUYÊN TẮC CƠ BẢN ĐỂ THIẾT KẾ
        """,
        "expected": {
            "document_code": "TCVN 4451:2012",
            "document_type": "original",
        }
    },
]


def test_metadata_extraction():
    print("=" * 60)
    print("Testing Document Metadata Extraction")
    print("=" * 60)
    
    extractor = DocumentMetadataExtractor(use_llm_fallback=False)
    
    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n--- Test Case {i} ---")
        result = extractor.extract(case["text"])
        
        print(f"Extracted: {result.to_dict()}")
        
        # Check expected values
        expected = case["expected"]
        for key, value in expected.items():
            actual = getattr(result, key)
            if actual == value:
                print(f"  ✓ {key}: {value}")
            else:
                print(f"  ✗ {key}: expected '{value}', got '{actual}'")
    
    print("\n" + "=" * 60)
    print("Testing complete!")


if __name__ == "__main__":
    test_metadata_extraction()
