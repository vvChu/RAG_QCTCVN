import pytest
from unittest.mock import MagicMock, patch
from src.domain.models import Chunk, RetrievalResult
from src.orchestrator.pipeline import RAGPipeline
from src.retrieval.hybrid import HybridRetriever
from src.generation.providers import GeminiProvider

@pytest.fixture
def mock_retriever():
    retriever = MagicMock(spec=HybridRetriever)
    chunk = Chunk(
        id="1", document_id="doc1", document_name="Test Doc",
        text="This is a test context.", page_number=1,
        chapter="Chapter 1", article="Article 1", clause="1"
    )
    result = RetrievalResult(chunk=chunk, score=0.9, rank=1)
    retriever.retrieve.return_value = [result]
    return retriever

@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=GeminiProvider)
    llm.generate.return_value = {
        "answer": "This is a test answer.",
        "model": "gemini-test",
        "prompt": "test prompt"
    }
    return llm

def test_rag_pipeline_success(mock_retriever, mock_llm):
    pipeline = RAGPipeline(mock_retriever, mock_llm)
    response = pipeline.query("test question")
    
    assert response.answer == "This is a test answer."
    assert len(response.contexts) == 1
    assert response.contexts[0].text == "This is a test context."
    assert response.stats.primary_model == "gemini-test"
    assert not response.stats.used_fallback

def test_rag_pipeline_fallback():
    # TODO: Implement fallback test
    pass
