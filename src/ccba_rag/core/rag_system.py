"""
CCBA RAG System - Main Facade

This is the primary entry point for using the RAG system programmatically.
It orchestrates all components: Embedder, VectorDB, Retriever, Generator.
"""

from functools import cached_property
from typing import Any, Dict, Optional

from ccba_rag.core.settings import settings
from ccba_rag.utils.logging import get_logger

logger = get_logger(__name__)


class RAGSystem:
    """
    Complete RAG System for Vietnamese Legal Documents.

    Service-oriented architecture:
    - Lazy-loaded components (embedder, vector_db, retriever, generators)
    - IndexingService for document ingestion
    - QueryService for query execution
    - Support for multiple LLM providers with fallback

    Usage:
        system = RAGSystem()
        result = system.query("Chiều cao tối thiểu của tầng 1?")
        print(result['answer'])
    """

    def __init__(self, config: Optional[Dict] = None, mode: str = "query", verbose: bool = True):
        """
        Initialize RAG System.

        Args:
            config: Optional config overrides
            mode: Operating mode ('query', 'index')
            verbose: Show detailed logs
        """
        self.config = config or {}
        self.mode = mode
        self.verbose = verbose

        # Lazy-loaded component caches
        self._embedder = None
        self._vector_db = None
        self._retriever = None
        self._reranker = None
        self._primary_generator = None
        self._fallback_generator = None
        self._chain = None

        if verbose:
            logger.info("RAG System initialized")

    @cached_property
    def embedder(self):
        """Get or create BGE-M3 embedder."""
        from ccba_rag.retrieval.embedder import BGEEmbedder

        logger.info("Initializing Embedder...")
        return BGEEmbedder()

    @cached_property
    def vector_db(self):
        """Get or create Milvus vector store."""
        from ccba_rag.retrieval.vectorstores.milvus import MilvusStore

        logger.info("Initializing Vector DB...")
        store = MilvusStore()
        # Ensure connected and collection loaded
        if store.has_collection():
            store.load_collection()
        return store

    @cached_property
    def reranker(self):
        """Get or create reranker (if enabled)."""
        if not settings.enable_reranker:
            return None

        from ccba_rag.retrieval.rerankers import BGEM3Reranker

        logger.info("Initializing Reranker...")
        return BGEM3Reranker()

    @cached_property
    def retriever(self):
        """Get or create hybrid retriever."""
        from ccba_rag.retrieval.retriever import HybridRetriever

        logger.info("Initializing Retriever...")
        return HybridRetriever(
            vector_db=self.vector_db, embedder=self.embedder, reranker=self.reranker
        )

    @cached_property
    def primary_generator(self):
        """Get or create primary generator."""
        from ccba_rag.generation.factory import create_generator

        logger.info("Initializing Primary Generator (Gemini)...")
        try:
            return create_generator("gemini")
        except Exception as e:
            logger.warning(f"Failed to create Gemini generator: {e}")
            return create_generator("groq")

    @cached_property
    def fallback_generator(self):
        """Get or create fallback generator."""
        from ccba_rag.generation.factory import create_generator

        logger.info("Initializing Fallback Generator (Groq)...")
        try:
            return create_generator("groq")
        except Exception as e:
            logger.warning(f"Failed to create fallback generator: {e}")
            return None

    @cached_property
    def chain(self):
        """Get or create RAG chain."""
        from ccba_rag.generation.chain import RAGChain

        logger.info("Initializing RAG Chain...")
        return RAGChain(
            retriever=self.retriever,
            primary_generator=self.primary_generator,
            fallback_generator=self.fallback_generator,
        )

    def query(self, question: str, verbose: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Execute a RAG query.

        Args:
            question: User question
            verbose: Show results
            **kwargs: Additional args (top_k, top_n, filters, etc.)

        Returns:
            Query result dict with answer, contexts, stats
        """
        result = self.chain.query(question, **kwargs)

        if verbose and self.verbose:
            self._display_results(result)

        return result

    def retrieve(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Retrieve contexts without generating answer.

        Args:
            question: User question
            **kwargs: Retrieval parameters

        Returns:
            Dict with contexts and retrieval stats
        """
        contexts, stats = self.retriever.retrieve(question, **kwargs)
        return {
            "query": question,
            "contexts": contexts,
            "stats": stats,
        }

    def index_documents(self, directory: str, drop_existing: bool = False):
        """
        Index documents from a directory.

        Args:
            directory: Path to documents directory
            drop_existing: Drop existing collection first
        """
        import asyncio

        from ccba_rag.ingestion.indexing_service import IndexingService

        logger.info(f"Starting indexing from: {directory}")

        # IndexingService creates its own components internally
        service = IndexingService(
            collection_name=settings.milvus_collection_name, chunk_size=1024, chunk_overlap=200
        )

        # Run the async index method
        asyncio.run(service.index_directory(directory, drop_existing=drop_existing))

    def _display_results(self, result: Dict):
        """Display query results in formatted output."""
        print("\n" + "=" * 60)
        print("ANSWER:")
        print(result.get("answer", "No answer"))
        print("\n" + "-" * 60)
        print("STATISTICS:")
        stats = result.get("stats", {})
        for key, value in stats.items():
            if "ms" in key:
                print(f"  {key}: {value:.1f}ms")
            else:
                print(f"  {key}: {value}")
        print(f"  model: {result.get('model')}")
        if result.get("used_fallback"):
            print("  ⚠️ Used fallback generator")
        print("=" * 60 + "\n")
