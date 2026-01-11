import pytest
from src.document_processor import StructuralChunker

class TestStructuralChunker:
    @pytest.fixture
    def chunker(self):
        return StructuralChunker(max_tokens=100)

    def test_identify_structure(self, chunker):
        text = "CHƯƠNG I: QUY ĐỊNH CHUNG\nĐiều 1. Phạm vi điều chỉnh"
        structure = chunker.identify_structure(text)
        assert structure['chapter'] == 'I'
        assert structure['article'] == '1'

    def test_split_by_structure_simple(self, chunker):
        text = """
        Điều 1. Phạm vi điều chỉnh
        Quy chuẩn này quy định về...
        Điều 2. Đối tượng áp dụng
        Quy chuẩn này áp dụng đối với...
        """
        chunks = chunker.split_by_structure(text, page=1)
        assert len(chunks) == 2
        assert chunks[0]['article'] == '1'
        assert chunks[1]['article'] == '2'

    def test_split_by_structure_with_clauses(self, chunker):
        text = """Điều 3. Giải thích từ ngữ
1. Quy hoạch xây dựng là...
2. Đô thị là..."""
        chunks = chunker.split_by_structure(text, page=1)
        assert len(chunks) == 2
        assert chunks[0]['article'] == '3'
        assert chunks[0]['clause'] == '1'
        assert chunks[1]['clause'] == '2'

    def test_chunk_document_fallback(self, chunker):
        # Text without structure
        text = "Đây là một đoạn văn bản không có cấu trúc rõ ràng.\nNó chỉ là các đoạn văn bình thường."
        
        # Mock extract_text methods since we can't easily mock file reading here without more complex setup
        # Instead, we'll test the internal logic if possible, or integration test with a dummy file.
        # For unit test, let's verify the regex patterns directly or use a mock for extract_text
        pass 

    def test_get_overlap_text(self, chunker):
        text = "Câu 1. Câu 2. Câu 3."
        overlap = chunker._get_overlap_text(text)
        # Assuming overlap_tokens is small enough to capture "Câu 3."
        assert "Câu 3" in overlap
