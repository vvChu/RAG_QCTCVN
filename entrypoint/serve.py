"""
CCBA RAG System - FastAPI Server

Production-ready API for the RAG system with:
- Query endpoint for RAG queries
- Index endpoint for document ingestion
- Status endpoint for health checks
- Static file serving for PDF rendering
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import os
import io
import time

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Import from new package location
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ccba_rag.core.settings import settings
from ccba_rag.core.models import QueryRequest, IndexRequest, FeedbackRequest
from ccba_rag.utils.logging import get_logger, configure_logging

configure_logging()
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CCBA RAG API",
    description="Vietnamese Construction Standards Expert System",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system instance
_rag_system = None


def get_rag_system():
    """Lazy-load RAG system."""
    global _rag_system
    if _rag_system is None:
        from ccba_rag.core.rag_system import RAGSystem
        logger.info("Initializing RAG System...")
        _rag_system = RAGSystem(verbose=True)
    return _rag_system


# ============================================================================
# API Models
# ============================================================================

class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    contexts: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}
    model: Optional[str] = None
    used_fallback: bool = False


class StatusResponse(BaseModel):
    """Response model for status endpoint."""
    status: str = "ok"
    collection: str
    milvus_host: str
    models: Dict[str, str]


class IndexResponse(BaseModel):
    """Response model for index endpoint."""
    status: str
    files_processed: int
    chunks_indexed: int


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "CCBA RAG API",
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/status", response_model=StatusResponse)
async def status():
    """Get system status."""
    return StatusResponse(
        status="ok",
        collection=settings.milvus_collection_name,
        milvus_host=settings.milvus_host,
        models={
            "embedding": settings.bge_model_name,
            "primary_llm": settings.gemini_model,
            "fallback_llm": settings.groq_model
        }
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Execute a RAG query.
    
    Args:
        request: Query parameters
        
    Returns:
        Answer with contexts and statistics
    """
    start_time = time.time()
    
    try:
        system = get_rag_system()
        
        result = system.query(
            question=request.query,
            verbose=False,
            top_k=request.top_k,
            top_n=request.top_n,
            history=request.history,
            filters=request.filters,
            use_reranker=request.use_reranker
        )
        
        # Clean contexts for serialization
        clean_contexts = []
        for ctx in result.get('contexts', []):
            clean_contexts.append({
                'text': ctx.get('text', ''),
                'document_name': ctx.get('document_name', ''),
                'article': ctx.get('article'),
                'chapter': ctx.get('chapter'),
                'clause': ctx.get('clause'),
                'score': ctx.get('rerank_score', ctx.get('retrieval_score', 0)),
            })
        
        return QueryResponse(
            answer=result.get('answer', 'No answer generated'),
            contexts=clean_contexts,
            stats=result.get('stats', {}),
            model=result.get('model'),
            used_fallback=result.get('used_fallback', False)
        )
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve")
async def retrieve(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(10, description="Number of results")
):
    """
    Retrieve contexts without generating answer.
    """
    try:
        system = get_rag_system()
        result = system.retrieve(query, top_k=top_k)
        return result
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index", response_model=IndexResponse)
async def index(request: IndexRequest):
    """
    Index documents from a directory.
    """
    try:
        system = get_rag_system()
        
        # Check directory exists
        if not Path(request.directory).exists():
            raise HTTPException(status_code=400, detail=f"Directory not found: {request.directory}")
        
        system.index_documents(
            directory=request.directory,
            drop_existing=request.drop_existing
        )
        
        return IndexResponse(
            status="success",
            files_processed=0,  # TODO: Return actual counts
            chunks_indexed=0
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    """
    Log user feedback for a query.
    """
    logger.info(f"Feedback: {request.rating} for query: {request.query[:50]}...")
    
    # TODO: Store feedback in database
    return {"status": "received", "rating": request.rating}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and index a single document.
    """
    # Save to temp directory
    temp_dir = Path("data/uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = temp_dir / file.filename
    
    try:
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Index the file
        system = get_rag_system()
        from ccba_rag.ingestion.indexing_service import IndexingService
        from ccba_rag.ingestion.splitters import StructuralSplitter
        
        service = IndexingService(
            chunker=StructuralSplitter(),
            embedder=system.embedder,
            vector_db=system.vector_db,
            verbose=True
        )
        
        chunks = service._process_file(file_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": len(chunks)
        }
    
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Static File Serving
# ============================================================================

# Mount static files if data directory exists
data_dir = Path("data")
if data_dir.exists():
    app.mount("/static", StaticFiles(directory=str(data_dir)), name="static")


@app.get("/pdf/{document_name}/page/{page_number}")
async def render_pdf_page(document_name: str, page_number: int):
    """
    Render a specific page of a PDF as PNG image.
    """
    try:
        import fitz
        
        # Find PDF in data directory
        pdf_path = None
        for path in data_dir.rglob(f"*{document_name}*"):
            if path.suffix.lower() == '.pdf':
                pdf_path = path
                break
        
        if not pdf_path:
            raise HTTPException(status_code=404, detail=f"PDF not found: {document_name}")
        
        doc = fitz.open(str(pdf_path))
        
        if page_number < 1 or page_number > len(doc):
            doc.close()
            raise HTTPException(status_code=400, detail=f"Invalid page number: {page_number}")
        
        page = doc[page_number - 1]
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        doc.close()
        
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF rendering failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup():
    """Pre-load RAG system on startup."""
    logger.info("Starting CCBA RAG API...")
    # Optionally pre-load: get_rag_system()


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Shutting down CCBA RAG API...")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "entrypoint.serve:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
