import time
from pathlib import Path
from typing import Any, Dict

import fitz

from ccba_rag.core.settings import settings
from ccba_rag.utils.logging import get_logger

from .base import ParserStrategy

logger = get_logger(__name__)

class GeminiVisionStrategy(ParserStrategy):
    """
    OCR fallback using Gemini 1.5 Flash Vision.
    Processes pages as images.
    """

    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse using Gemini Vision OCR."""
        try:
            import google.generativeai as genai

            api_key = settings.gemini_api_key
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found")

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('models/gemini-2.0-flash')

            doc = fitz.open(file_path)
            full_text = []
            pages_data = []

            logger.info(f"Starting Gemini Vision OCR for {file_path.name} ({len(doc)} pages)")

            for page_num, page in enumerate(doc, 1):
                # Convert page to image (pixmap)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # 2x zoom for better resolution
                img_data = pix.tobytes("png")

                # Create image part
                image_part = {
                    "mime_type": "image/png",
                    "data": img_data
                }

                retry_count = 0
                page_text = ""

                while retry_count < 3:
                    try:
                        response = model.generate_content(
                            [
                                "Transcription of this document page. Output ONLY the text content. Preserve Vietnamese characters accurately.",
                                image_part
                            ],
                            request_options={"timeout": 60}
                        )
                        page_text = response.text
                        break
                    except Exception as e:
                        retry_count += 1
                        logger.warning(f"Gemini OCR failed for page {page_num} (Attempt {retry_count}): {e}")
                        time.sleep(2 * retry_count)

                if not page_text:
                    logger.error(f"Failed to OCR page {page_num} after 3 attempts")
                    page_text = "[OCR FAILED]"

                pages_data.append({
                    'page_number': page_num,
                    'text': page_text,
                    'char_count': len(page_text)
                })
                full_text.append(page_text)

                # Rate limit handling (simple sleep)
                time.sleep(1)

            doc.close()
            combined_text = "\n".join(full_text)

            return {
                'text': combined_text,
                'pages': pages_data,
                'stats': {
                    'parser': 'gemini_vision',
                    'total_pages': len(pages_data),
                    'total_chars': len(combined_text)
                },
                'metadata': {
                    'file_path': str(file_path),
                    'file_name': file_path.name,
                    'file_size': file_path.stat().st_size
                }
            }

        except Exception as e:
            logger.error(f"Gemini Vision failed for {file_path.name}: {e}")
            raise
