"""
Milvus/Zilliz Cloud Vector Store Implementation

Supports both local Milvus and Zilliz Cloud with hybrid search (Dense + Sparse).
Uses Reciprocal Rank Fusion (RRF) for combining search results.
"""

import json
from typing import Dict, List, Optional

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from ccba_rag.core.base import BaseVectorDB
from ccba_rag.core.models import Chunk, ChunkLevel, RetrievalResult
from ccba_rag.core.settings import settings
from ccba_rag.utils.logging import get_logger

logger = get_logger(__name__)


class MilvusStore(BaseVectorDB):
    """
    Milvus/Zilliz Cloud vector database implementation.

    Features:
    - Hybrid search (Dense HNSW + Sparse Inverted Index)
    - RRF fusion for combining results
    - Automatic connection management
    - Support for both local Milvus and Zilliz Cloud
    """

    def __init__(self, collection_name: Optional[str] = None):
        """
        Initialize MilvusStore.

        Args:
            collection_name: Override collection name from settings
        """
        self.alias = "default"
        self._connected = False
        self.collection_name = collection_name or settings.milvus_collection_name
        self.collection: Optional[Collection] = None

    def _ensure_connection(self) -> None:
        """Ensure we are connected to Milvus/Zilliz."""
        if self._connected:
            return

        try:
            if connections.has_connection(self.alias):
                self._connected = True
                return

            connection_args = {"alias": self.alias}

            if settings.milvus_secure:
                # Zilliz Cloud connection
                connection_args.update({
                    "uri": f"https://{settings.milvus_host}:{settings.milvus_port}",
                    "secure": True,
                })
            else:
                # Local Milvus connection
                connection_args.update({
                    "host": settings.milvus_host,
                    "port": settings.milvus_port,
                })

            if settings.milvus_user and settings.milvus_password:
                connection_args["user"] = settings.milvus_user
                connection_args["password"] = settings.milvus_password

            connections.connect(**connection_args)
            self._connected = True
            logger.info(f"Connected to Milvus at {settings.milvus_host}:{settings.milvus_port}")

        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def has_collection(self) -> bool:
        """Check if the collection exists."""
        self._ensure_connection()
        return utility.has_collection(self.collection_name)

    def create_collection(self, dim: int = 1024, drop_if_exists: bool = False) -> None:
        """
        Create collection with schema for legal documents.

        Args:
            dim: Vector dimension (1024 for BGE-M3)
            drop_if_exists: Drop existing collection first
        """
        self._ensure_connection()

        if drop_if_exists and utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"Dropped existing collection: {self.collection_name}")

        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
            return

        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="document_name", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="chapter", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="article", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="clause", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="level", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="full_context", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="bbox", dtype=DataType.VARCHAR, max_length=200),
        ]

        schema = CollectionSchema(fields=fields, description="CCBA Legal Documents")
        self.collection = Collection(name=self.collection_name, schema=schema)

        # Create Dense Vector Index (HNSW)
        dense_index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {
                "M": settings.milvus_hnsw_m,
                "efConstruction": settings.milvus_hnsw_ef_construction
            }
        }
        self.collection.create_index(field_name="dense_vector", index_params=dense_index_params)

        # Create Sparse Vector Index
        sparse_index_params = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
            "params": {"drop_ratio_build": 0.2}
        }
        self.collection.create_index(field_name="sparse_vector", index_params=sparse_index_params)

        logger.info(f"Created collection '{self.collection_name}' with HNSW + Sparse indexes")

    def create_index(self) -> None:
        """Create indexes (called separately if needed)."""
        # Already handled in create_collection
        pass

    def load_collection(self) -> None:
        """Load collection into memory for searching."""
        self._ensure_connection()
        if self.collection is None:
            self.collection = Collection(self.collection_name)
        self.collection.load()

    def insert(self, chunks: List[Chunk]) -> List[str]:
        """
        Insert chunks into the collection.

        Args:
            chunks: List of Chunk objects with vectors populated

        Returns:
            List of inserted chunk IDs
        """
        self._ensure_connection()
        if self.collection is None:
            self.collection = Collection(self.collection_name)

        # Prepare data with truncation for safety
        data = [
            [chunk.id for chunk in chunks],
            [chunk.dense_vector for chunk in chunks],
            [chunk.sparse_vector for chunk in chunks],
            [chunk.document_id[:100] for chunk in chunks],
            [chunk.document_name[:500] for chunk in chunks],
            [(chunk.chapter or "")[:500] for chunk in chunks],
            [(chunk.article or "")[:500] for chunk in chunks],
            [(chunk.clause or "")[:500] for chunk in chunks],
            [chunk.text[:20000] for chunk in chunks],  # Truncate long texts
            [chunk.page_number for chunk in chunks],
            [chunk.parent_id or "" for chunk in chunks],
            [chunk.level.value if isinstance(chunk.level, ChunkLevel) else str(chunk.level) for chunk in chunks],
            [(chunk.full_context or "")[:65535] for chunk in chunks],
            [json.dumps(chunk.bbox) if chunk.bbox else "" for chunk in chunks],
        ]

        self.collection.insert(data)
        self.collection.flush()
        logger.info(f"Inserted {len(chunks)} chunks into Milvus")
        return [chunk.id for chunk in chunks]

    def search(
        self,
        query_dense: List[float],
        query_sparse: Dict[int, float],
        top_k: int = 100,
        expr: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Perform hybrid search with RRF fusion.

        Args:
            query_dense: Dense query vector
            query_sparse: Sparse query vector (token_id -> weight)
            top_k: Number of results to return
            expr: Optional filter expression

        Returns:
            List of RetrievalResult sorted by RRF score
        """
        self._ensure_connection()
        if self.collection is None:
            self.collection = Collection(self.collection_name)

        self.collection.load()

        output_fields = [
            "document_id", "document_name", "chapter", "article", "clause",
            "text", "page_number", "parent_id", "level", "full_context", "bbox"
        ]

        # Dense Search
        dense_search_params = {
            "metric_type": "COSINE",
            "params": {"ef": settings.milvus_hnsw_ef_search}
        }
        dense_results = self.collection.search(
            data=[query_dense],
            anns_field="dense_vector",
            param=dense_search_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields
        )

        # Sparse Search
        sparse_search_params = {"metric_type": "IP", "params": {}}
        sparse_results = self.collection.search(
            data=[query_sparse],
            anns_field="sparse_vector",
            param=sparse_search_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields
        )

        # RRF Fusion
        k = 60  # RRF constant
        rrf_scores: Dict[str, float] = {}
        chunks_map: Dict[str, Chunk] = {}

        def process_hits(hits):
            for rank, hit in enumerate(hits):
                doc_id = str(hit.id)
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                rrf_scores[doc_id] += 1 / (k + rank + 1)

                if doc_id not in chunks_map:
                    # Parse bbox from JSON string
                    bbox_str = hit.entity.get('bbox', '')
                    bbox = None
                    if bbox_str:
                        try:
                            bbox = json.loads(bbox_str)
                        except:
                            pass

                    # Parse level
                    level_str = hit.entity.get('level', 'clause')
                    try:
                        level = ChunkLevel(level_str)
                    except:
                        level = ChunkLevel.CLAUSE

                    chunks_map[doc_id] = Chunk(
                        id=doc_id,
                        document_id=hit.entity.get('document_id', ''),
                        document_name=hit.entity.get('document_name', ''),
                        text=hit.entity.get('text', ''),
                        page_number=hit.entity.get('page_number', 1),
                        chapter=hit.entity.get('chapter'),
                        article=hit.entity.get('article'),
                        clause=hit.entity.get('clause'),
                        parent_id=hit.entity.get('parent_id'),
                        level=level,
                        full_context=hit.entity.get('full_context'),
                        bbox=bbox
                    )

        if dense_results:
            process_hits(dense_results[0])
        if sparse_results:
            process_hits(sparse_results[0])

        # Sort by RRF score and build results
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        final_results = []
        for i, uid in enumerate(sorted_ids[:top_k]):
            chunk = chunks_map[uid]
            final_results.append(RetrievalResult(chunk=chunk, score=rrf_scores[uid], rank=i + 1))

        return final_results

    def delete_collection(self) -> None:
        """Delete the collection."""
        self._ensure_connection()
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")

    def delete_by_document(self, document_name: str) -> int:
        """
        Delete all chunks for a specific document.

        Args:
            document_name: Document name to delete

        Returns:
            Number of deleted chunks (approximate)
        """
        self._ensure_connection()
        if not self.has_collection():
            return 0

        if self.collection is None:
            self.collection = Collection(self.collection_name)

        self.collection.load()
        expr = f'document_name == "{document_name}"'
        self.collection.delete(expr)
        logger.info(f"Deleted chunks for document: {document_name}")
        return 1  # Milvus doesn't return count directly
