# Project Structure

```
RAG_QCTCVN/
├── .env                          # Environment configuration
├── .env.example                  # Template for .env
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── docker-compose.yml            # Milvus setup
├── README.md                     # Main documentation
│
├── src/                          # Source code
│   ├── document_processor.py    # PDF parsing & chunking
│   ├── embedder.py               # BGE-M3 embedding
│   ├── vector_store.py           # Milvus integration
│   ├── reranker.py               # Vietnamese Reranker
│   ├── generator.py              # Gemini Pro CLI
│   ├── evaluation.py             # Metrics & evaluation
│   └── main.py                   # Main orchestration
│
├── docs/                         # Documentation
│   ├── CONSTITUTION.md           # 4 core principles
│   ├── SPECIFICATION.md          # Technical specification
│   ├── QUICKSTART.md             # Quick start guide
│   ├── API_REFERENCE.md          # API documentation
│   └── DEPLOYMENT.md             # Deployment guide
│
├── examples/                     # Example scripts
│   ├── simple_usage.py           # Basic usage example
│   └── evaluation_example.py    # Evaluation example
│
├── data/                         # Data directory
│   └── pdfs/                     # PDF documents (user-provided)
│
├── tests/                        # Unit tests (TODO)
│   └── test_*.py
│
└── config/                       # Configuration files
    └── (optional custom configs)
```

## File Descriptions

### Core Source Files

#### `src/document_processor.py`
- **Purpose**: Parse PDF và structural chunking
- **Key Classes**: `StructuralChunker`, `DocumentChunk`
- **Principle**: Nguyên tắc 3 - Phân đoạn theo cấu trúc logic

#### `src/embedder.py`
- **Purpose**: BGE-M3 embedding với Multi-Functionality
- **Key Classes**: `BGEEmbedder`, `HybridSearchScorer`
- **Principle**: Nguyên tắc 2 - Hybrid Retrieval

#### `src/vector_store.py`
- **Purpose**: Milvus vector database integration
- **Key Classes**: `MilvusVectorDB`
- **Features**: HNSW index, hybrid search support

#### `src/reranker.py`
- **Purpose**: Vietnamese Cross-encoder reranking
- **Key Classes**: `VietnameseReranker`, `TwoStageRetriever`
- **Performance**: Top 100 → Top 5, < 50ms

#### `src/generator.py`
- **Purpose**: Gemini Pro generation với grounding
- **Key Classes**: `GeminiGenerator`, `RAGPipeline`
- **Principle**: Nguyên tắc 1 & 4 - R+G strategy, citations

#### `src/evaluation.py`
- **Purpose**: Metrics & evaluation framework
- **Key Classes**: `RetrievalEvaluator`, `FaithfulnessEvaluator`, `RAGEvaluator`
- **Metrics**: nDCG@k, MRR@k, Recall@k, Faithfulness

#### `src/main.py`
- **Purpose**: Main orchestration & CLI
- **Key Classes**: `RAGSystem`
- **Usage**: Index, Query, Evaluate modes

### Documentation Files

#### `docs/CONSTITUTION.md`
4 Nguyên tắc cốt lõi không thể thương lượng:
1. Tính chính xác thực tế tuyệt đối
2. Hybrid Retrieval bắt buộc
3. Phân đoạn theo cấu trúc logic
4. Trích dẫn nguồn chi tiết

#### `docs/SPECIFICATION.md`
Đặc tả kỹ thuật đầy đủ:
- System architecture & flow
- Component specifications
- Performance requirements
- Evaluation framework

#### `docs/QUICKSTART.md`
Hướng dẫn bắt đầu nhanh (5 bước):
1. Setup environment
2. Install dependencies
3. Configure
4. Index documents
5. Query

#### `docs/API_REFERENCE.md`
API documentation:
- All public classes & methods
- Parameters & return types
- Usage examples
- Configuration options

#### `docs/DEPLOYMENT.md`
Production deployment guide:
- Full installation checklist
- Performance tuning
- Troubleshooting
- Monitoring & scaling

### Configuration Files

#### `.env` (user-created from `.env.example`)
Runtime configuration:
- Milvus connection
- Model names
- API keys
- Hyperparameters

#### `requirements.txt`
Python dependencies:
- Core: FlagEmbedding, transformers, torch
- Vector DB: pymilvus
- LLM: google-generativeai
- Utils: numpy, pandas, tqdm

#### `docker-compose.yml`
Milvus stack:
- Milvus standalone
- etcd (metadata)
- MinIO (storage)

### Example Scripts

#### `examples/simple_usage.py`
Basic usage:
- Initialize system
- Index documents
- Run queries

#### `examples/evaluation_example.py`
Evaluation workflow:
- Load test dataset
- Run batch queries
- Calculate metrics
- Aggregate results

## Module Dependencies

```
main.py
├── document_processor.py
├── embedder.py
│   └── FlagEmbedding (external)
├── vector_store.py
│   └── pymilvus (external)
├── reranker.py
│   ├── embedder.py
│   └── transformers (external)
├── generator.py
│   └── google.generativeai (external)
└── evaluation.py
    └── numpy (external)
```

## Data Flow

```
1. PDF → document_processor → Chunks
                                  ↓
2. Chunks → embedder → Dense/Sparse Vectors
                                  ↓
3. Vectors → vector_store (Milvus) → Indexed
                                  ↓
4. Query → embedder → Query Vectors
                                  ↓
5. Query Vectors → vector_store → Top 100 Candidates
                                  ↓
6. Candidates → reranker → Top 5 Contexts
                                  ↓
7. Query + Top 5 → generator → Final Answer with Citations
```

## Key Design Decisions

### 1. Modular Architecture
- Each component is independent
- Easy to swap implementations
- Clear separation of concerns

### 2. Configuration-driven
- All parameters in .env
- Easy to tune without code changes
- Support multiple environments

### 3. Evaluation-first
- Built-in metrics from day 1
- Continuous quality monitoring
- Data-driven optimization

### 4. Documentation-complete
- Every module documented
- Clear examples provided
- Production-ready guides

## Next Steps for Development

### Short-term (Weeks 1-2)
- [ ] Add unit tests (tests/)
- [ ] Implement sparse search in Milvus
- [ ] Add LLM-as-Judge faithfulness
- [ ] Web UI với Gradio

### Mid-term (Weeks 3-4)
- [ ] Support DOCX, HTML formats
- [ ] Multi-language support
- [ ] Batch processing optimization
- [ ] Caching layer

### Long-term (Months 2-3)
- [ ] RAG over RAG (hierarchical)
- [ ] Active learning pipeline
- [ ] Model fine-tuning tools
- [ ] Production monitoring dashboard

## Testing Strategy

### Unit Tests
```python
tests/
├── test_document_processor.py
├── test_embedder.py
├── test_vector_store.py
├── test_reranker.py
├── test_generator.py
└── test_evaluation.py
```

### Integration Tests
```python
tests/integration/
├── test_indexing_pipeline.py
├── test_query_pipeline.py
└── test_end_to_end.py
```

### Performance Tests
```python
tests/performance/
├── test_latency.py
├── test_throughput.py
└── test_resource_usage.py
```

## Contributing Guidelines

1. **Code Style**: Follow PEP 8
2. **Documentation**: Docstrings for all public functions
3. **Testing**: Add tests for new features
4. **Constitution**: Always respect 4 core principles
5. **Spec-Driven**: Update SPECIFICATION.md when changing architecture
