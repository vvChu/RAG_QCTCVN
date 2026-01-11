from pathlib import Path
from typing import Any, Dict, List, Optional

from ccba_rag.utils.logging import get_logger

from .base import ParserStrategy
from .gemini_vision import GeminiVisionStrategy
from .llama_parse import LlamaParseStrategy
from .pymupdf import PyMuPDFStrategy

logger = get_logger(__name__)

class DocumentParser:
    """
    Factory and orchestrator for parsing strategies.
    Implements the fallback logic.
    """

    def __init__(self, strategies: List[ParserStrategy] = None):
        """
        Initialize with a list of strategies to try in order.
        If None, uses default chain: LlamaParse -> PyMuPDF -> GeminiVision
        """
        if strategies is None:
            self.strategies = [
                LlamaParseStrategy(), # High quality, handles tables/scans
                PyMuPDFStrategy(),    # Fast, fallback for simple stats
                GeminiVisionStrategy() # Last resort OCR
            ]
        else:
            self.strategies = strategies

    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parse file using strategies in order until one succeeds.
        """
        path = Path(file_path)
        errors = []

        for strategy in self.strategies:
            strategy_name = strategy.__class__.__name__
            try:
                # Basic optimization: Don't try PyMuPDF on strictly scanned docs if we can know ahead?
                # For now, just try in order.

                # Special skip: If LlamaParse fails, PyMuPDF might return garbage for scanned files.
                # But PyMuPDFStrategy returns stats, enabling detection of "scanned-ness".

                logger.info(f"Parser: Trying {strategy_name} for {path.name}")
                result = strategy.parse(path)

                if result:
                    logger.info(f"Parser: {strategy_name} succeeded")
                    return result

            except Exception as e:
                error_msg = f"{strategy_name} failed: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                continue

        # If we get here, all failed
        raise RuntimeError(f"All parsing strategies failed for {path.name}. Errors: {errors}")

__all__ = [
    'ParserStrategy',
    'PyMuPDFStrategy',
    'LlamaParseStrategy',
    'GeminiVisionStrategy',
    'DocumentParser'
]
