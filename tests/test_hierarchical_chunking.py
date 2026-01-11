import unittest
import sys
import os
from unittest.mock import MagicMock

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.document_processor import StructuralChunker
from src.domain.models import ChunkLevel

class TestHierarchicalChunking(unittest.TestCase):
    
    def setUp(self):
        self.chunker = StructuralChunker()
        self.document_id = "doc_123"
        self.document_name = "Luật Test"

    def test_chunking_structure(self):
        text = """
        CHƯƠNG I: QUY ĐỊNH CHUNG
        
        Điều 1. Phạm vi điều chỉnh
        Luật này quy định về việc thử nghiệm.
        
        Điều 2. Đối tượng áp dụng
        1. Cơ quan nhà nước.
        2. Tổ chức, cá nhân.
        """
        
        chunks = self.chunker._parse_structure(text, self.document_id, self.document_name)
        
        # Verify we have chunks
        self.assertTrue(len(chunks) > 0)
        
        # Check for Article 1 (No clauses, so it might be Clause level or Article level depending on logic)
        # In my logic: Article Chunk (Parent) + Clause Chunk (if clauses exist) OR Simple Chunk
        # For "Điều 1", it has no "1." pattern, so it falls back to _create_simple_chunks inside _process_clauses?
        # Let's trace: _process_articles -> Article Chunk created -> _process_clauses
        # _process_clauses -> no match -> _create_simple_chunks -> returns 1 chunk
        
        # So we expect:
        # 1. Article 1 Parent Chunk
        # 2. Article 1 Content Chunk (Level CLAUSE or DOCUMENT? Logic says CLAUSE if article_title is present)
        # 3. Article 2 Parent Chunk
        # 4. Clause 1 Chunk
        # 5. Clause 2 Chunk
        
        article_chunks = [c for c in chunks if c.level == ChunkLevel.ARTICLE]
        clause_chunks = [c for c in chunks if c.level == ChunkLevel.CLAUSE]
        
        self.assertEqual(len(article_chunks), 2, "Should have 2 Article chunks")
        self.assertEqual(len(clause_chunks), 3, "Should have 3 Clause/Content chunks (Art 1 content + Art 2 Clause 1 + Art 2 Clause 2)")
        
        # Verify Context
        art1_content = clause_chunks[0]
        self.assertIn("Luật Test > CHƯƠNG I: QUY ĐỊNH CHUNG > Điều 1. Phạm vi điều chỉnh", art1_content.full_context)
        
        clause1 = clause_chunks[1]
        self.assertEqual(clause1.clause, "1.")
        self.assertIn("Luật Test > CHƯƠNG I: QUY ĐỊNH CHUNG > Điều 2. Đối tượng áp dụng", clause1.full_context)
        self.assertEqual(clause1.parent_id, article_chunks[1].id)

    def test_chunking_no_chapters(self):
        text = """
        Điều 1. Test No Chapter
        Nội dung điều 1.
        """
        chunks = self.chunker._parse_structure(text, self.document_id, self.document_name)
        
        self.assertTrue(len(chunks) > 0)
        first_chunk = chunks[1] # Index 0 is Article Parent
        self.assertIn("Luật Test >", first_chunk.full_context)
        self.assertNotIn("Chương", first_chunk.full_context)

if __name__ == '__main__':
    unittest.main()
