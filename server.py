"""
CCBA RAG API Server (Legacy Compatible)

This server uses the new ccba_rag package while maintaining
API compatibility with the existing Streamlit frontend (app.py).
"""

import logging
import os
import sys
import shutil
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Query
from fastapi.responses import FileResponse, Response
import fitz  # PyMuPDF
import io
import json

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import from new package
from ccba_rag.core.rag_system import RAGSystem
from ccba_rag.core.settings import settings
from ccba_rag.utils.logging import get_logger
from ccba_rag.core.models import QueryRequest, FeedbackRequest, IndexRequest

logger = get_logger(__name__)


# --- MODELS ---
# Imported from ccba_rag.core.models


# --- GLOBAL STATE ---
rag_system: Optional[RAGSystem] = None


def log_feedback(query: str, answer: str, rating: str, model: str):
    """Log feedback to file."""
    import json
    from datetime import datetime
    
    feedback_dir = Path("data/feedback")
    feedback_dir.mkdir(parents=True, exist_ok=True)
    
    feedback_file = feedback_dir / "feedback.jsonl"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer[:500],
        "rating": rating,
        "model": model
    }
    
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global rag_system
    print("🚀 Initializing RAG System... (Loading Models)")
    try:
        rag_system = RAGSystem(mode='query', verbose=True)
        
        # Pre-load components
        _ = rag_system.embedder
        _ = rag_system.vector_db
        
        print("✅ RAG System Ready!")
    except Exception as e:
        print(f"❌ Failed to initialize RAG System: {e}")
        raise e
        
    yield
    
    # Shutdown
    print("🛑 Shutting down RAG System...")


app = FastAPI(title="CCBA Expert API", lifespan=lifespan)


# --- HELPER TASKS ---
def run_indexing_task(directory: str):
    print(f"Started indexing task for {directory}")
    rag_system.index_documents(directory)
    print("Indexing completed")


# --- ENDPOINTS ---

@app.get("/")
async def root():
    """Root endpoint to show server status."""
    return {
        "message": "RAG API Server is running!",
        "docs_url": "/docs",
        "ui_url": "http://localhost:8501"
    }


@app.get("/status")
async def status():
    if rag_system:
        # Get count if available
        count = 0
        try:
            if hasattr(rag_system.vector_db, 'collection') and rag_system.vector_db.collection:
                 count = rag_system.vector_db.collection.num_entities
        except:
             pass
             
        return {
            "status": "ready", 
            "collection": settings.milvus_collection_name,
            "stats": {
                "chunk_count": count
            }
        }
    return {"status": "initializing"}


@app.get("/documents")
async def get_documents():
    """Get list of indexed documents"""
    docs = []
    for ext in ["*.pdf", "*.docx"]:
        for p in Path("data/documents").rglob(ext):
            docs.append(p.name)
    return {"documents": sorted(list(set(docs)))}


@app.get("/files/{filename}")
async def get_file(filename: str):
    """Serve document files safely"""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    base_path = Path("data/documents")
    found_files = list(base_path.rglob(filename))
    
    if not found_files:
        found_files = list(base_path.rglob(f"{filename}.*"))
        
    if not found_files:
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    file_path = found_files[0]
    media_type = "application/pdf" if file_path.suffix.lower() == ".pdf" else None
        
    return FileResponse(file_path, media_type=media_type)


@app.get("/render/{filename}/{page}")
async def render_page(filename: str, page: int, bbox: str = Query(None)):
    """Render PDF page as PNG with optional highlight"""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    base_path = Path("data/documents")
    found_files = list(base_path.rglob(filename))
    if not found_files:
        found_files = list(base_path.rglob(f"{filename}.*"))
        
    if not found_files:
        raise HTTPException(status_code=404, detail="File not found")
        
    file_path = found_files[0]
    if file_path.suffix.lower() != '.pdf':
        raise HTTPException(status_code=400, detail="Only PDF rendering supported")
        
    try:
        doc = fitz.open(file_path)
        if page < 1 or page > len(doc):
            raise HTTPException(status_code=400, detail="Page out of range")
            
        pdf_page = doc[page - 1]
        
        if bbox:
            try:
                coords = [float(x) for x in bbox.split(',')]
                if len(coords) == 4:
                    rect = fitz.Rect(coords)
                    annot = pdf_page.add_highlight_annot(rect)
                    annot.update()
            except Exception as e:
                print(f"Highlight error: {e}")
                
        pix = pdf_page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")
        
        return Response(content=img_bytes, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Render error: {e}")


@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a file and trigger indexing for it."""
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    allowed_types = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    if file.content_type not in allowed_types and not file.filename.endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are allowed")
        
    upload_dir = Path("data/documents/uploaded")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
        
    background_tasks.add_task(run_indexing_task, "data/documents")
    
    return {"message": f"File '{file.filename}' uploaded. Indexing started in background."}


@app.post("/index")
async def index_endpoint(request: IndexRequest, background_tasks: BackgroundTasks):
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not os.path.exists(request.directory):
        raise HTTPException(status_code=404, detail="Directory not found")
        
    background_tasks.add_task(run_indexing_task, request.directory)
    return {"message": f"Indexing started for {request.directory} in background"}


@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    try:
        log_feedback(request.query, request.answer, request.rating, request.model)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Execute Query
        result = rag_system.query(
            question=request.query,
            verbose=False,
            top_k=request.top_k,
            top_n=request.top_n,
            history=request.history,
            filters=request.filters,
            use_reranker=request.use_reranker
        )
        
        # Format Output for frontend compatibility
        contexts = result.get('contexts', [])
        output_data = {
            "original_query": request.query,
            "detected_intent": "construction_legal_advice",
            "answer": result.get('answer', 'No answer'),
            "citations": [
                {
                    "content": c.get('text', '')[:200],
                    "document": c.get('document_name', 'Unknown'),
                    "section": f"Điều {c.get('article', '?')}, Khoản {c.get('clause', '?')}",
                    "page": c.get('page_number', 1),
                    "bbox": c.get('bbox')
                }
                for c in contexts[:5]
            ],
            "stats": {
                "model": result.get('model', 'unknown'),
                "retrieval_ms": result.get('stats', {}).get('retrieval_ms', 0),
                "total_ms": result.get('stats', {}).get('total_ms', 0)
            },
            "prompt": result.get('prompt', '')
        }
        return output_data
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)