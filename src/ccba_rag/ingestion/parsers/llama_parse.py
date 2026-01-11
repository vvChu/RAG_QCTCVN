import os
from pathlib import Path
from typing import Any, Dict, List

import nest_asyncio

from ccba_rag.core.settings import settings
from ccba_rag.utils.logging import get_logger

from .base import ParserStrategy

logger = get_logger(__name__)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

class LlamaParseStrategy(ParserStrategy):
    """
    Advanced parsing using LlamaParse API.
    Essential for scanned PDFs and complex tables.
    """

    def __init__(self, mode: str = "balanced"):
        self.mode = mode

    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF using LlamaParse."""
        try:
            from llama_parse import LlamaParse

            # Use KeyManager for rotation
            from ccba_rag.utils.key_manager import llama_key_manager

            available_keys = llama_key_manager.get_keys()
            if not available_keys:
                raise ValueError("No LLAMA_CLOUD_API_KEY found")

            last_exception = None

            # Try keys in order (rotation handled by manager if we used get_next_key,
            # but simpler here to iterate available list so we don't skip fresh keys)
            # Actually, let's use the list directly to be robust.

            for key_idx, api_key in enumerate(available_keys):
                try:
                    logger.info(f"LlamaParse: Using key ...{api_key[-4:]} (Attempt {key_idx+1}/{len(available_keys)})")

                    parser = LlamaParse(
                        api_key=api_key,
                        result_type="markdown",
                        num_workers=4,
                        verbose=True,
                        language="vi",
                        premium_mode=True
                    )

                    logger.info(f"Sending {file_path.name} to LlamaParse...")
                    documents = parser.load_data(str(file_path))

                    if not documents:
                        raise ValueError("LlamaParse returned no content")

                    # Combine text and preserve page structure
                    full_text = []
                    pages_data = []

                    for i, doc in enumerate(documents, 1):
                        page_text = doc.text
                        full_text.append(page_text)

                        pages_data.append({
                            'page_number': i,
                            'text': page_text,
                            'char_count': len(page_text)
                        })

                    combined_text = "\n\n".join(full_text)

                    if not combined_text or not combined_text.strip():
                         raise ValueError("LlamaParse returned empty text")

                    return {
                        'text': combined_text,
                        'pages': pages_data,
                        'stats': {
                            'parser': 'llamaparse',
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
                    logger.warning(f"LlamaParse failed with key ...{api_key[-4:]}: {e}")
                    last_exception = e
                    # Continue to next key

            # If all keys failed
            if last_exception:
                raise last_exception

        except Exception as e:
            logger.error(f"LlamaParseStrategy failed for {file_path.name}: {e}")
            raise
