"""
Test Embedder Module

Tests BGE-M3 embedding functionality.
Note: These tests require the BGE-M3 model to be downloaded.
Run with: pytest tests/test_embedder.py -v
"""

import pytest
import numpy as np


class TestBGEEmbedder:
    """Tests for BGE-M3 embedder."""
    
    @pytest.fixture(scope="class")
    def embedder(self):
        """Load embedder once for all tests in class."""
        from ccba_rag.retrieval.embedder import BGEEmbedder
        return BGEEmbedder()
    
    def test_embedder_initialization(self, embedder):
        """Test that embedder initializes correctly."""
        assert embedder is not None
        assert embedder.model is not None
    
    def test_embedding_dimension(self, embedder):
        """Test that embedding dimension is correct."""
        dim = embedder.get_embedding_dim()
        assert dim == 1024
    
    def test_encode_single_query(self, embedder):
        """Test encoding a single query."""
        result = embedder.encode_queries("What is the building code?")
        
        assert 'dense' in result
        assert 'sparse' in result
        assert result['dense'].shape == (1, 1024)
    
    def test_encode_multiple_queries(self, embedder):
        """Test encoding multiple queries."""
        queries = [
            "Building safety regulations",
            "Fire code requirements"
        ]
        result = embedder.encode_queries(queries)
        
        assert result['dense'].shape == (2, 1024)
        assert len(result['sparse']) == 2
    
    def test_encode_documents(self, embedder):
        """Test encoding documents."""
        docs = ["This is a test document about construction."]
        result = embedder.encode_documents(docs)
        
        assert result['dense'].shape == (1, 1024)
    
    def test_encode_all_returns_lists(self, embedder):
        """Test that encode_all returns list format."""
        texts = ["Test text 1", "Test text 2"]
        dense, sparse = embedder.encode_all(texts, show_progress=False)
        
        assert isinstance(dense, list)
        assert isinstance(sparse, list)
        assert len(dense) == 2
        assert len(sparse) == 2
    
    def test_singleton_pattern(self):
        """Test that embedder follows singleton pattern."""
        from ccba_rag.retrieval.embedder import BGEEmbedder
        
        e1 = BGEEmbedder()
        e2 = BGEEmbedder()
        
        assert e1 is e2  # Same instance


class TestHybridSearchScorer:
    """Tests for RRF score fusion."""
    
    def test_rrf_fusion(self):
        """Test Reciprocal Rank Fusion."""
        from ccba_rag.retrieval.embedder import HybridSearchScorer
        
        scorer = HybridSearchScorer(k=60)
        
        dense_results = [
            {'id': 'doc1', 'score': 0.9},
            {'id': 'doc2', 'score': 0.8},
            {'id': 'doc3', 'score': 0.7},
        ]
        
        sparse_results = [
            {'id': 'doc2', 'score': 0.95},
            {'id': 'doc1', 'score': 0.85},
            {'id': 'doc4', 'score': 0.75},
        ]
        
        fused = scorer.fuse_results(dense_results, sparse_results)
        
        # doc1 and doc2 should be at top (appear in both)
        top_ids = [r['id'] for r in fused[:2]]
        assert 'doc1' in top_ids
        assert 'doc2' in top_ids
        
        # All docs should have rrf_score
        for doc in fused:
            assert 'rrf_score' in doc
