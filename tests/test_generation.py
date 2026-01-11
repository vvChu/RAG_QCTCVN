import pytest
from unittest.mock import MagicMock, patch
from src.generation.gemini_generator import GeminiGenerator
from src.generation.groq_generator import GroqGenerator

@pytest.fixture
def mock_gemini_response():
    mock = MagicMock()
    mock.text = "Gemini Answer"
    return mock

@patch('google.generativeai.GenerativeModel')
def test_gemini_generator(mock_model_class, mock_gemini_response):
    # Setup mock
    mock_model_instance = mock_model_class.return_value
    mock_model_instance.generate_content.return_value = mock_gemini_response
    
    # Initialize
    generator = GeminiGenerator(api_key="test_key")
    
    # Test generate
    result = generator.generate(
        query="Test Query",
        contexts=[{'text': 'Context 1', 'document_name': 'Doc 1'}]
    )
    
    assert result['answer'] == "Gemini Answer"
    assert result['model'] == "gemini-2.0-flash"
    assert result['error'] is None

@patch('src.generation.groq_generator.Groq')
def test_groq_generator(mock_groq_class):
    # Setup mock
    mock_client = mock_groq_class.return_value
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "Groq Answer"
    mock_client.chat.completions.create.return_value = mock_completion
    
    # Initialize
    generator = GroqGenerator(api_key="test_key")
    
    # Test generate
    result = generator.generate(
        query="Test Query",
        contexts=[{'text': 'Context 1', 'document_name': 'Doc 1'}]
    )
    
    assert result['answer'] == "Groq Answer"
    assert result['model'] == "llama-3.3-70b-versatile"
    assert "error" not in result or result['error'] is None
