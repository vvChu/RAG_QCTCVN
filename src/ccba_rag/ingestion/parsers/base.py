from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class ParserStrategy(ABC):
    """
    Abstract base class for document parsing strategies.

    Each strategy implements a method to parse a file and return
    standardized dictionary containing text, metadata, and pages.
    """

    @abstractmethod
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse a document file.

        Args:
            file_path: Path to the file to parse

        Returns:
            Dict containing:
            - text: Full text content
            - pages: List of page contents/metadata
            - stats: Parsing statistics (page count, etc.)
            - metadata: File metadata

            Returns None or raises Exception if parsing fails.
        """
        pass
