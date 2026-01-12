"""
Document Metadata Extractor

Automatically extracts document metadata from content:
- Document code (QCVN/TCVN number)
- Document type (original/amendment/supplement)
- Related documents (amends, replaces, references)
- Effective date
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from ccba_rag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentMetadata:
    """Extracted document metadata."""
    document_code: Optional[str] = None  # e.g., "QCVN 06:2022/BXD"
    document_type: str = "original"  # original, amendment, supplement
    amendment_number: Optional[str] = None  # e.g., "1:2023"
    amends_document: Optional[str] = None  # Document being amended
    replaces_document: Optional[str] = None  # Document being replaced
    references: List[str] = None  # Referenced documents
    effective_date: Optional[str] = None  # Effective date
    issuing_body: Optional[str] = None  # BXD, BNNPTNT, etc.

    def __post_init__(self):
        if self.references is None:
            self.references = []

    def to_dict(self) -> dict:
        return {
            'document_code': self.document_code,
            'document_type': self.document_type,
            'amendment_number': self.amendment_number,
            'amends_document': self.amends_document,
            'replaces_document': self.replaces_document,
            'references': self.references,
            'effective_date': self.effective_date,
            'issuing_body': self.issuing_body,
        }


class DocumentMetadataExtractor:
    """
    Extract document metadata from Vietnamese legal document content.

    Uses regex patterns first for speed, falls back to LLM for complex cases.
    """

    # Regex patterns for Vietnamese legal documents
    PATTERNS = {
        # QCVN 06:2022/BXD or TCVN 4451:2012
        'document_code': re.compile(
            r'(QCVN|TCVN)\s*(\d+)\s*[:\-]\s*(\d{4})(?:/(\w+))?',
            re.IGNORECASE
        ),

        # Sửa đổi 1:2023 của QCVN 06:2022/BXD
        'amendment': re.compile(
            r'[Ss]ửa\s+đổi\s+(\d+)\s*[:\-]\s*(\d{4})',
            re.IGNORECASE
        ),

        # "của QCVN XX:YYYY" - document being amended
        'amends': re.compile(
            r'(?:của|cho)\s+(QCVN|TCVN)\s*(\d+)\s*[:\-]\s*(\d{4})(?:/(\w+))?',
            re.IGNORECASE
        ),

        # Thay thế QCVN XX:YYYY
        'replaces': re.compile(
            r'[Tt]hay\s+thế\s+(QCVN|TCVN)\s*(\d+)\s*[:\-]\s*(\d{4})',
            re.IGNORECASE
        ),

        # Ban hành theo Quyết định số XX/YYYY/QD-BXD
        'issued_by': re.compile(
            r'[Qq]uyết\s+định\s+số\s*([\d/\-]+[A-Z\-]+)',
            re.IGNORECASE
        ),

        # Có hiệu lực từ ngày DD/MM/YYYY or DD-MM-YYYY
        'effective_date': re.compile(
            r'[Hh]iệu\s+lực\s+(?:từ\s+ngày|kể\s+từ)\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})',
            re.IGNORECASE
        ),

        # References: theo TCVN XXXX, căn cứ QCVN XX
        'references': re.compile(
            r'(?:theo|căn\s+cứ|tham\s+chiếu)\s+((?:QCVN|TCVN)\s*\d+\s*[:\-]\s*\d{4})',
            re.IGNORECASE
        ),
    }

    def __init__(self, use_llm_fallback: bool = True):
        """
        Initialize extractor.

        Args:
            use_llm_fallback: Use LLM for complex cases when regex fails
        """
        self.use_llm_fallback = use_llm_fallback

    def extract(self, text: str, filename: str = "") -> DocumentMetadata:
        """
        Extract metadata from document text.

        Args:
            text: Full document text (or first ~2000 chars for efficiency)
            filename: Original filename for additional hints

        Returns:
            DocumentMetadata object
        """
        # Use first 3000 chars for metadata extraction
        header_text = text[:3000] if len(text) > 3000 else text

        metadata = DocumentMetadata()

        # 1. Extract document code
        code_match = self.PATTERNS['document_code'].search(header_text)
        if code_match:
            doc_type = code_match.group(1).upper()
            number = code_match.group(2)
            year = code_match.group(3)
            body = code_match.group(4) or ""

            if body:
                metadata.document_code = f"{doc_type} {number}:{year}/{body}"
            else:
                metadata.document_code = f"{doc_type} {number}:{year}"

            metadata.issuing_body = body

        # 2. Check if it's an amendment
        amend_match = self.PATTERNS['amendment'].search(header_text)
        if amend_match:
            metadata.document_type = "amendment"
            amend_num = amend_match.group(1)
            amend_year = amend_match.group(2)
            metadata.amendment_number = f"{amend_num}:{amend_year}"

            # Find what document is being amended
            amends_match = self.PATTERNS['amends'].search(header_text)
            if amends_match:
                doc_type = amends_match.group(1).upper()
                number = amends_match.group(2)
                year = amends_match.group(3)
                body = amends_match.group(4) or ""

                if body:
                    metadata.amends_document = f"{doc_type} {number}:{year}/{body}"
                else:
                    metadata.amends_document = f"{doc_type} {number}:{year}"

        # 3. Check if it replaces another document
        replaces_match = self.PATTERNS['replaces'].search(header_text)
        if replaces_match:
            doc_type = replaces_match.group(1).upper()
            number = replaces_match.group(2)
            year = replaces_match.group(3)
            metadata.replaces_document = f"{doc_type} {number}:{year}"

        # 4. Extract effective date
        date_match = self.PATTERNS['effective_date'].search(header_text)
        if date_match:
            metadata.effective_date = date_match.group(1)

        # 5. Extract references
        references = self.PATTERNS['references'].findall(header_text)
        if references:
            metadata.references = list(set(references))

        # 6. If incomplete and LLM enabled, use LLM fallback
        if self.use_llm_fallback and not metadata.document_code:
            llm_metadata = self._extract_with_llm(header_text)
            if llm_metadata:
                metadata = self._merge_metadata(metadata, llm_metadata)

        # Log extraction result
        if metadata.document_code:
            logger.info(f"Extracted metadata: {metadata.document_code} ({metadata.document_type})")
            if metadata.amends_document:
                logger.info(f"  └─ Amends: {metadata.amends_document}")
        else:
            logger.warning(f"Could not extract document code from: {filename}")

        return metadata

    def _extract_with_llm(self, text: str) -> Optional[DocumentMetadata]:
        """Use LLM to extract metadata when regex fails."""
        try:
            from ccba_rag.generation.factory import create_generator

            prompt = """Phân tích đoạn văn bản quy chuẩn/tiêu chuẩn Việt Nam và trích xuất thông tin sau:

1. document_code: Mã văn bản (VD: "QCVN 06:2022/BXD" hoặc "TCVN 4451:2012")
2. document_type: "original" (bản gốc) hoặc "amendment" (sửa đổi/bổ sung)
3. amendment_number: Số hiệu sửa đổi nếu có (VD: "1:2023")
4. amends_document: Mã văn bản đang được sửa đổi (nếu là bản sửa đổi)

Văn bản:
---
{text}
---

Trả về CHÍNH XÁC JSON format sau (không giải thích):
{{"document_code": "...", "document_type": "...", "amendment_number": "...", "amends_document": "..."}}
"""

            generator = create_generator("groq")  # Use fast model
            response = generator.generate(
                prompt.format(text=text[:1500]),
                max_tokens=200,
                temperature=0
            )

            # Parse JSON response
            import json
            # Try to find JSON in response
            json_match = re.search(r'\{[^}]+\}', response.text)
            if json_match:
                data = json.loads(json_match.group())
                return DocumentMetadata(
                    document_code=data.get('document_code'),
                    document_type=data.get('document_type', 'original'),
                    amendment_number=data.get('amendment_number'),
                    amends_document=data.get('amends_document')
                )

        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")

        return None

    def _merge_metadata(
        self,
        primary: DocumentMetadata,
        secondary: DocumentMetadata
    ) -> DocumentMetadata:
        """Merge two metadata objects, preferring primary values."""
        return DocumentMetadata(
            document_code=primary.document_code or secondary.document_code,
            document_type=primary.document_type if primary.document_type != "original" else secondary.document_type,
            amendment_number=primary.amendment_number or secondary.amendment_number,
            amends_document=primary.amends_document or secondary.amends_document,
            replaces_document=primary.replaces_document or secondary.replaces_document,
            references=primary.references or secondary.references,
            effective_date=primary.effective_date or secondary.effective_date,
            issuing_body=primary.issuing_body or secondary.issuing_body,
        )


# Convenience function
def extract_document_metadata(text: str, filename: str = "") -> Dict:
    """Extract metadata from document text."""
    extractor = DocumentMetadataExtractor(use_llm_fallback=False)  # Regex only for speed
    return extractor.extract(text, filename).to_dict()
