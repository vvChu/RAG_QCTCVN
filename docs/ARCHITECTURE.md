# CCBA RAG System Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Entry Points                              │
├─────────────┬─────────────┬─────────────┬─────────────────────────┤
│  server.py  │   app.py    │  app_ui.py  │ entrypoint/cli.py       │
│  (FastAPI)  │ (Streamlit) │  (Gradio)   │     (Typer CLI)         │
└──────┬──────┴──────┬──────┴──────┬──────┴──────────┬──────────────┘
       │             │             │                  │
       └─────────────┴─────────────┴──────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │     RAGSystem       │
                    │  (Facade Pattern)   │
                    └──────────┬──────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       │                       │                       │
┌──────▼──────┐        ┌───────▼───────┐       ┌───────▼───────┐
│  Ingestion  │        │   Retrieval   │       │  Generation   │
├─────────────┤        ├───────────────┤       ├───────────────┤
│ loaders.py  │        │  embedder.py  │       │  chain.py     │
│ splitters.py│        │  retriever.py │       │  factory.py   │
│ indexing_   │        │  rerankers.py │       │  generators/  │
│  service.py │        │  vectorstores/│       │   gemini.py   │
└─────────────┘        └───────────────┘       │   groq.py     │
                                               │   deepseek.py │
                                               └───────────────┘
```

## Module Descriptions

### Core (`src/ccba_rag/core/`)

| File | Purpose |
|------|---------|
| `rag_system.py` | Main facade with lazy-loaded components |
| `settings.py` | Pydantic settings + YAML loader |
| `prompts.py` | Prompt template management |
| `models.py` | Pydantic domain models (Chunk, RAGResponse) |
| `constants.py` | Regex patterns, defaults |
| `base.py` | Abstract base classes |

### Ingestion (`src/ccba_rag/ingestion/`)

| File | Purpose |
|------|---------|
| `loaders.py` | PDF/DOCX document loaders |
| `splitters.py` | Structural text chunking |
| `indexing_service.py` | Document indexing pipeline |

### Retrieval (`src/ccba_rag/retrieval/`)

| File | Purpose |
|------|---------|
| `embedder.py` | BGE-M3 dense+sparse embeddings |
| `retriever.py` | Hybrid retrieval with RRF |
| `rerankers.py` | Cross-encoder reranking |
| `vectorstores/milvus.py` | Milvus/Zilliz connector |

### Generation (`src/ccba_rag/generation/`)

| File | Purpose |
|------|---------|
| `chain.py` | RAG chain with fallback |
| `factory.py` | Generator factory |
| `generators/*.py` | LLM implementations |

## Data Flow

```
Document → Loader → Splitter → Embedder → Milvus
                                            │
Query → Embedder → Hybrid Search ───────────┘
                         │
                         ▼
               Reranker (optional)
                         │
                         ▼
                    Generator
                         │
                         ▼
                      Answer
```

## Configuration

| File | Purpose |
|------|---------|
| `config/default.yaml` | Model parameters |
| `config/prompts.yaml` | System prompts |
| `.env` | API keys and credentials |
