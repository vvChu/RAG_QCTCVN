from typing import Dict, Any, List
from pathlib import Path
import fitz  # PyMuPDF
import re
from .base import ParserStrategy
from ccba_rag.utils.logging import get_logger

logger = get_logger(__name__)

class PyMuPDFStrategy(ParserStrategy):
    """
    Fast, local PDF parsing using PyMuPDF (fitz).
    Best for digital-native PDFs.
    """
    
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF using PyMuPDF."""
        try:
            doc = fitz.open(file_path)
            full_text = []
            pages_data = []
            
            total_tables_detected = 0
            total_images_detected = 0
            
            for page_num, page in enumerate(doc, 1):
                text = page.get_text("text")
                
                # Basic heuristics for stats
                tables = len(re.findall(r'(?:\|.*\|)|(?:\+[-+]+\+)', text))
                images = len(page.get_images())
                
                total_tables_detected += tables
                total_images_detected += images
                
                page_info = {
                    'page_number': page_num,
                    'text': text,
                    'char_count': len(text),
                    'tables_detected': tables,
                    'images_detected': images
                }
                
                pages_data.append(page_info)
                full_text.append(text)
                
            doc.close()
            
            combined_text = "\n".join(full_text)
            
            if not combined_text or not combined_text.strip():
                 raise ValueError("PyMuPDF extracted no text (likely scanned)")
            
            return {
                'text': combined_text,
                'pages': pages_data,
                'stats': {
                    'parser': 'pymupdf',
                    'total_pages': len(pages_data),
                    'total_chars': len(combined_text),
                    'total_tables': total_tables_detected,
                    'total_images': total_images_detected
                },
                'metadata': {
                    'file_path': str(file_path),
                    'file_name': file_path.name,
                    'file_size': file_path.stat().st_size
                }
            }
            
        except Exception as e:
            logger.error(f"PyMuPDF failed for {file_path.name}: {e}")
            raise
