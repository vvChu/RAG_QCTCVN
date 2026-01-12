"""
Constants for the CCBA RAG System

This module centralizes all hardcoded values including:
- Regex patterns for Vietnamese legal document structure recognition
- Model parameters and defaults
- Schema field limits
"""

import re

# =============================================================================
# DOCUMENT STRUCTURE PATTERNS (Vietnamese Legal Documents)
# =============================================================================

# Chapter: "CHƯƠNG I", "Chương 1", "CHƯƠNG I:", "Chương I - Quy định chung"
CHAPTER_PATTERN = re.compile(
    r'(?:CHƯƠNG|Chương)\s+([IVXLCDM]+|[0-9]+)[\s:.\-]*(.+)?',
    re.IGNORECASE
)

# Article: "ĐIỀU 1", "Điều 1.", "Điều 1: Phạm vi áp dụng"
ARTICLE_PATTERN = re.compile(
    r'(?:ĐIỀU|Điều)\s+([0-9]+)[\s:.\-]*(.+)?',
    re.IGNORECASE
)

# Clause: "1.", "2.", etc. at start of line
CLAUSE_PATTERN = re.compile(
    r'^\s*([0-9]+)[\s.]+',
    re.MULTILINE
)

# Section Level 1: "1. ", "2. " (e.g., "2. QUY ĐỊNH KỸ THUẬT")
SECTION_LEVEL_1_PATTERN = re.compile(r'^\s*(\d+)\.\s+(.+)', re.MULTILINE)

# Section Level 2: "1.1", "1.2" (e.g., "1.1. Phạm vi áp dụng")
SECTION_LEVEL_2_PATTERN = re.compile(r'^\s*(\d+\.\d+)\.?\s+(.+)', re.MULTILINE)

# Section Level 3: "1.1.1", "1.1.2" (e.g., "1.1.1. Nhiệt độ")
SECTION_LEVEL_3_PATTERN = re.compile(r'^\s*(\d+\.\d+\.\d+)\.?\s+(.+)', re.MULTILINE)

# Appendix: "PHỤ LỤC A", "Phụ lục 1"
APPENDIX_PATTERN = re.compile(
    r'(?:PHỤ LỤC|Phụ lục)\s+([0-9A-Z]+)[\s:.\-]*(.*)?',
    re.IGNORECASE
)

# Helper pattern to extract numbers
NUMBER_PATTERN = re.compile(r'\d+')

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

# BGE-M3 Embedder defaults
BGE_DEFAULT_MODEL = "BAAI/bge-m3"
BGE_DEFAULT_MAX_LENGTH = 1024  # CPU optimized (vs 8192 for GPU)
BGE_DEFAULT_BATCH_SIZE = 64   # CPU optimized
BGE_EMBEDDING_DIM = 1024

# Reranker defaults
RERANKER_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_DEFAULT_BATCH_SIZE = 64
RERANKER_DEFAULT_TOP_K = 100
RERANKER_DEFAULT_TOP_N = 5

# HNSW Index defaults (CPU optimized)
HNSW_DEFAULT_M = 8
HNSW_DEFAULT_EF_CONSTRUCTION = 100
HNSW_DEFAULT_EF_SEARCH = 32

# RRF (Reciprocal Rank Fusion)
RRF_DEFAULT_K = 60

# Generation defaults
GENERATION_DEFAULT_TEMPERATURE = 0.1
GENERATION_DEFAULT_MAX_TOKENS = 1024

# =============================================================================
# COLLECTION & DATABASE
# =============================================================================

DEFAULT_COLLECTION_NAME = "legal_documents"
DEFAULT_MILVUS_HOST = "localhost"
DEFAULT_MILVUS_PORT = "19530"

# Schema field limits (Milvus VARCHAR constraints)
MAX_VARCHAR_LENGTH_SHORT = 500   # document_id, chapter, article, clause
MAX_VARCHAR_LENGTH_MEDIUM = 500  # document_name
MAX_VARCHAR_LENGTH_LONG = 65535  # text content

# =============================================================================
# INGESTION
# =============================================================================

MAX_CHUNK_CHARS = 2000
MIN_CHUNK_CHARS = 100
CHUNK_OVERLAP_CHARS = 200

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc"}

# =============================================================================
# NOISE PATTERNS (for text cleaning)
# =============================================================================

NOISE_PATTERNS = [
    r'Trang\s+\d+\s*/\s*\d+',           # Page numbers: "Trang 1 / 10"
    r'Page\s+\d+\s+of\s+\d+',            # English page numbers
    r'^\d+$',                             # Standalone numbers
    r'QCVN\s+\d+:\d+/BXD',               # Repeated header references
    r'www\.[^\s]+',                       # URLs
    r'http[s]?://[^\s]+',                # URLs with protocol
]

# Compile patterns for efficiency
COMPILED_NOISE_PATTERNS = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in NOISE_PATTERNS]
