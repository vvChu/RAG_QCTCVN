# Technical Specification - RAG System for Vietnamese Legal Documents

## 1. System Overview

### 1.1 Purpose
Xây dựng hệ thống Retrieval-Augmented Generation (RAG) chuyên sâu cho kho dữ liệu PDF Quy chuẩn và Tiêu chuẩn Xây dựng Việt Nam, đảm bảo tính chính xác thực tế tuyệt đối và khả năng trích dẫn nguồn.

### 1.2 Key Requirements
- Tìm kiếm kết hợp: Từ khóa chính xác + Ngữ nghĩa phức tạp
- Câu trả lời có căn cứ (Grounded) và trích dẫn nguồn chi tiết
- Hỗ trợ tiếng Việt đầy đủ
- Xử lý văn bản pháp luật có cấu trúc phức tạp

## 2. Architecture

### 2.1 System Flow
```
PDF Documents → Parsing → Structural Chunking → BGE-M3 Encoding
                                                      ↓
User Query → BGE-M3 Encoding → Milvus Hybrid Search (Dense+Sparse)
                                                      ↓
                                        Top 100 Candidates
                                                      ↓
                                    Vietnamese Reranker (Cross-encoder)
                                                      ↓
                                          Top 5 Contexts
                                                      ↓
                                    Gemini Pro CLI Generation
                                                      ↓
                              Final Answer with Citations
```

### 2.2 Components

#### A. Embedding & Retrieval Model
- **Model**: BAAI/bge-m3
- **Type**: Multi-Functionality (Dense + Sparse)
- **Dimension**: 1024
- **Max Sequence Length**: 8192 tokens
- **Features**:
  - Dense embeddings for semantic search
  - Sparse lexical weights for keyword matching
  - Multi-Granularity processing

#### B. Vector Database
- **System**: Milvus
- **Index Type**: HNSW (Hierarchical Navigable Small World)
- **Reason**: Tối ưu QPS với độ giảm NDCG@10 ≤ 0.005
- **Collections**: 
  - Dense vectors (1024 dim)
  - Sparse weights
  - Metadata (document, chapter, article, clause)

#### C. Reranking Model
- **Model**: Vietnamese_Reranker
- **Type**: Cross-encoder (fine-tuned from BGE-reranker-v2-m3)
- **Performance**: 0.7944 Accuracy@1 on Legal Zalo 2021
- **Parameters**:
  - Input: Top K = 100
  - Output: Top N = 5
  - Latency: < 50ms

#### D. Generation Model
- **Model**: Gemini Pro CLI
- **Strategy**: Retrieval-Augmented Generation (R+G)
- **Constraints**:
  - Only answer based on provided context
  - Must include citations
  - Grounding to source documents

## 3. Technical Specifications

### 3.1 Data Processing

#### PDF Parsing
- Libraries: `pdfplumber` + `pypdf2`
- Extract: Text, structure, metadata
- Preserve: Document hierarchy

#### Structural Chunking
- Strategy: Logic-based (Chapter → Article → Clause)
- Max chunk size: 8192 tokens
- Overlap: Minimal (only at logical boundaries)
- Metadata retention:
  ```python
  {
    "document_id": str,
    "document_name": str,
    "chapter": str,
    "article": str,
    "clause": str,
    "text": str,
    "token_count": int
  }
  ```

### 3.2 Embedding & Indexing

#### BGE-M3 Configuration
```python
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
output = model.encode(
    texts,
    return_dense=True,
    return_sparse=True,
    max_length=8192
)
```

#### Milvus Schema
```python
schema = {
    "fields": [
        {"name": "id", "type": "INT64", "is_primary": True},
        {"name": "dense_vector", "type": "FLOAT_VECTOR", "dim": 1024},
        {"name": "sparse_vector", "type": "SPARSE_FLOAT_VECTOR"},
        {"name": "metadata", "type": "JSON"}
    ]
}

# HNSW Index for dense vectors
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200}
}
```

### 3.3 Retrieval & Reranking

#### Hybrid Retrieval
- Dense weight: 0.5
- Sparse weight: 0.5
- Top K: 100
- Search type: ANN (Approximate Nearest Neighbor)

#### Reranking Process
```python
reranker = CrossEncoder('Vietnamese_Reranker')
scores = reranker.predict([(query, doc) for doc in top_100])
top_5 = sorted(zip(scores, top_100), reverse=True)[:5]
```

### 3.4 Generation

#### Prompt Template
```
Bạn là trợ lý AI chuyên về quy chuẩn xây dựng Việt Nam.

NGUYÊN TẮC:
1. CHỈ trả lời dựa trên ngữ cảnh được cung cấp bên dưới
2. Trích dẫn nguồn chính xác: [Tên văn bản - Điều X, Khoản Y]
3. Nếu không tìm thấy thông tin, nói rõ "Không tìm thấy trong tài liệu"

NGỮ CẢNH:
{context_1}
...
{context_5}

CÂU HỎI: {user_query}

TRẢ LỜI:
```

## 4. Performance Requirements

### 4.1 Retrieval Quality
- nDCG@10: ≥ 0.85
- MRR@10: ≥ 0.80
- Recall@100: ≥ 0.95

### 4.2 System Performance
- End-to-end latency: < 3 seconds
- Retrieval latency: < 100ms
- Reranking latency: < 50ms
- Generation latency: < 2 seconds

### 4.3 Accuracy
- Faithfulness: ≥ 0.90 (measured by LLM-as-Judge)
- Citation accuracy: 100% (all claims must be cited)

## 5. Evaluation Framework

### 5.1 Retrieval Metrics
- nDCG@k (k=1,5,10)
- MRR@k (k=1,5,10)
- Recall@k (k=10,50,100)

### 5.2 Generation Metrics
- Faithfulness (LLM-as-Judge)
- Citation coverage
- Answer relevance

### 5.3 End-to-End Metrics
- User satisfaction
- Task completion rate
- Error rate

## 6. Deployment Considerations

### 6.1 Infrastructure
- Milvus: Docker deployment or Zilliz Cloud
- Models: GPU-enabled servers (CUDA)
- API: RESTful or gRPC

### 6.2 Scalability
- Horizontal scaling for Milvus
- Model serving with batching
- Caching for frequent queries

### 6.3 Monitoring
- Query latency distribution
- Retrieval quality metrics
- Error tracking and alerting
