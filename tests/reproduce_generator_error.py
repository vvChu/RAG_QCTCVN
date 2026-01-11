
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))
from unittest.mock import MagicMock, patch
from src.generator import GeminiGenerator

def test_generator_error_handling():
    """
    Test that the generator returns a dictionary with the 'model' key
    even when an exception occurs.
    """
    print("Testing Generator Error Handling...")
    
    # Mock environment variable if not present
    if 'GEMINI_API_KEY' not in os.environ:
        os.environ['GEMINI_API_KEY'] = 'test_key'
        
    generator = GeminiGenerator(api_key='test_key', model='test-model')
    
    # Mock _call_gemini_cli to raise an exception
    with patch.object(generator, '_call_gemini_cli', side_effect=Exception("Simulated API Error")):
        result = generator.generate("test query", [{"text": "context", "document_name": "doc"}])
        
        print(f"Result keys: {list(result.keys())}")
        
        if 'model' in result:
            print("SUCCESS: 'model' key found in result.")
            print(f"Model value: {result['model']}")
        else:
            print("FAILURE: 'model' key MISSING in result.")
            sys.exit(1)

if __name__ == "__main__":
    try:
        test_generator_error_handling()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
