import hashlib
import re
from typing import Optional

def normalize_document_code(code: str) -> str:
    """
    Normalize document code to a consistent ID format.
    
    Examples:
        "QCVN 06:2022/BXD" -> "qcvn_06_2022_bxd"
        "TCVN 4451:2012" -> "tcvn_4451_2012"
    """
    if not code:
        return ""
    
    # Lowercase
    normalized = code.lower()
    
    # Replace special chars with underscore
    normalized = re.sub(r'[:\-/\\s]+', '_', normalized)
    
    # Remove multiple underscores
    normalized = re.sub(r'_+', '_', normalized)
    
    # Remove trailing underscores
    normalized = normalized.strip('_')
    
    return normalized

def generate_document_id(file_path: str, document_code: str = None) -> str:
    """
    Generate a unique document ID.
    
    If document_code is provided, use normalized code.
    Otherwise, fall back to file path hash.
    """
    if document_code:
        normalized = normalize_document_code(document_code)
        if normalized:
            return normalized
    
    # Fallback to path hash
    return hashlib.md5(file_path.encode()).hexdigest()[:16]
