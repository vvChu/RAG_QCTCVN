
import requests
import time
from typing import Dict, Any, List, Optional
from ccba_rag.utils.logging import get_logger

logger = get_logger(__name__)

class APIClient:
    """
    Client for interacting with the CCBA RAG API Server.
    Encapsulates endpoint urls, timeout logic, and error handling.
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Parse JSON or raise error."""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"API Error ({response.status_code}): {response.text}")
            raise e
        except Exception as e:
            logger.error(f"Request Error: {e}")
            raise e

    def get_status(self) -> Dict[str, Any]:
        """Check server status."""
        resp = requests.get(f"{self.base_url}/status", timeout=2)
        return self._handle_response(resp)

    def get_documents(self) -> List[str]:
        """Get list of indexed documents."""
        resp = requests.get(f"{self.base_url}/documents", timeout=5)
        data = self._handle_response(resp)
        return data.get("documents", [])

    def query(self, 
              query: str, 
              history: List[Dict] = None, 
              filters: Dict = None,
              top_k: int = 20, 
              top_n: int = 5,
              use_expansion: bool = False,
              use_reranker: bool = True
             ) -> Dict[str, Any]:
        """Send a query to the RAG system."""
        payload = {
            "query": query,
            "top_k": top_k,
            "top_n": top_n,
            "use_expansion": use_expansion,
            "use_reranker": use_reranker,
            "history": history or []
        }
        if filters:
            payload["filters"] = filters
            
        resp = requests.post(f"{self.base_url}/query", json=payload, timeout=300)
        return self._handle_response(resp)

    def upload_file(self, file_obj, filename: str, content_type: str) -> Dict[str, Any]:
        """Upload a file for indexing."""
        files = {"file": (filename, file_obj, content_type)}
        resp = requests.post(f"{self.base_url}/upload", files=files, timeout=300)
        return self._handle_response(resp)

    def trigger_indexing(self, directory: str = "data/documents") -> Dict[str, Any]:
        """Trigger background indexing."""
        resp = requests.post(f"{self.base_url}/index", json={"directory": directory}, timeout=10)
        return self._handle_response(resp)

    def send_feedback(self, query: str, answer: str, rating: str, model: str) -> None:
        """Send user feedback."""
        payload = {
            "query": query,
            "answer": answer,
            "rating": rating,
            "model": model
        }
        requests.post(f"{self.base_url}/feedback", json=payload, timeout=5)

    def get_file_url(self, filename: str, page: int = 1) -> str:
        """Construct a URL for direct file access."""
        return f"{self.base_url}/files/{filename}#page={page}"

