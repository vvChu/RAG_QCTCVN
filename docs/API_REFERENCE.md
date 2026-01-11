# API Reference

## Core Classes

### RAGSystem

Main orchestration class.

```python
from src.main import RAGSystem

system = RAGSystem(config=None)
```

**Parameters:**
- `config` (dict, optional): Configuration override

**Methods:**

#### `index_documents(pdf_dir: str)`

Index PDF documents from directory.

```python
system.index_documents("data/pdfs")
```

#### `query(question: str, verbose: bool = True) -> dict`

Query the RAG system.

```python
result = system.query("Your question here?")
```

**Returns:**
```python
{
    'answer': str,           # Generated answer
    'contexts': List[Dict],  # Top N contexts
    'citations': List[Dict], # Citation metadata
    'stats': Dict            # Performance stats
}
```

---

### BGEEmbedder

BGE-M3 embedding model wrapper.

```python
from src.embedder import BGEEmbedder

embedder = BGEEmbedder(
    model_name='BAAI/bge-m3',
    use_fp16=True,
    max_length=8192
)
```

**Methods:**

#### `encode_queries(queries: Union[str, List[str]]) -> Dict`

Encode queries to dense/sparse embeddings.

```python
embeddings = embedder.encode_queries("Your query")
# Returns: {'dense': np.ndarray, 'sparse': List[Dict]}
```

#### `encode_documents(documents: Union[str, List[str]]) -> Dict`

Encode documents to dense/sparse embeddings.

---

### MilvusVectorDB

Milvus vector database interface.

```python
from src.vector_store import MilvusVectorDB

# Local Milvus
db = MilvusVectorDB(
    host="localhost",
    port="19530",
    collection_name="legal_documents"
)

# Zilliz Cloud
db = MilvusVectorDB(
    host="your-endpoint.vectordb.zillizcloud.com",
    port="19530",
    user="db_74b5693bc1c4c80",
    password="your_password",
    secure=True,
    collection_name="legal_documents"
)
```

**Methods:**

#### `create_collection(dense_dim: int = 1024, drop_if_exists: bool = False)`

Create collection with schema.

#### `create_index()`

Create HNSW index for dense vectors.

#### `insert_documents(dense_vectors, sparse_vectors, metadata) -> List[int]`

Insert documents and return IDs.

#### `hybrid_search(query_dense, query_sparse, top_k=100) -> List[Dict]`

Perform hybrid search.

---

### VietnameseReranker

Cross-encoder reranker.

```python
from src.reranker import VietnameseReranker

reranker = VietnameseReranker(
    model_name="Vietnamese_Reranker"
)
```

**Methods:**

#### `rerank(query: str, documents: List[Dict], top_n: int = 5) -> List[Dict]`

Rerank documents by relevance.

```python
ranked = reranker.rerank(
    query="Your query",
    documents=candidates,
    top_n=5
)
```

---

### GeminiGenerator

Gemini Pro integration for generation.

```python
from src.generator import GeminiGenerator

generator = GeminiGenerator(
    api_key="your_key",
    model="gemini-pro"
)
```

**Methods:**

#### `generate(query: str, contexts: List[Dict], temperature: float = 0.1) -> Dict`

Generate answer with citations.

```python
result = generator.generate(
    query="Your question",
    contexts=top_contexts
)
```

---

### RAGEvaluator

Evaluation framework.

```python
from src.evaluation import RAGEvaluator

evaluator = RAGEvaluator()
```

**Methods:**

#### `evaluate_query(query_result: Dict, ground_truth_docs: List[str]) -> Dict`

Evaluate single query result.

```python
eval_result = evaluator.evaluate_query(
    query_result=result,
    ground_truth_docs=['doc1', 'doc2']
)
```

**Returns:**
```python
{
    'retrieval_metrics': {
        'nDCG@10': float,
        'MRR@10': float,
        'Recall@100': float
    },
    'generation_metrics': {
        'citation_coverage': float,
        'faithfulness_score': float
    }
}
```

#### `evaluate_batch(query_results: List[Dict], ground_truths: List[List[str]]) -> Dict`

Evaluate multiple queries and aggregate.

---

## Configuration

### Environment Variables

All config in `.env`:

```env
# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=legal_documents

# BGE-M3
BGE_MODEL_NAME=BAAI/bge-m3
BGE_MAX_LENGTH=8192
BGE_USE_FP16=True

# Reranker
RERANKER_MODEL_NAME=Vietnamese_Reranker
RERANKER_TOP_K=100
RERANKER_TOP_N=5

# Gemini
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-pro

# Hybrid Search
HYBRID_DENSE_WEIGHT=0.5
HYBRID_SPARSE_WEIGHT=0.5
```

### Programmatic Config

```python
config = {
    'bge_max_length': 4096,
    'reranker_top_k': 50,
    'reranker_top_n': 3,
    'hybrid_dense_weight': 0.7,
    'hybrid_sparse_weight': 0.3,
}

system = RAGSystem(config=config)
```

---

## Data Structures

### DocumentChunk

```python
@dataclass
class DocumentChunk:
    document_id: str
    document_name: str
    chapter: Optional[str]
    article: Optional[str]
    clause: Optional[str]
    text: str
    token_count: int
    page_number: int
```

### Query Result

```python
{
    'answer': str,
    'contexts': [
        {
            'id': int,
            'document_id': str,
            'document_name': str,
            'chapter': str,
            'article': str,
            'clause': str,
            'text': str,
            'rerank_score': float
        },
        ...
    ],
    'citations': [
        {
            'document_name': str,
            'article': str,
            'clause': str,
            'text_preview': str
        },
        ...
    ],
    'stats': {
        'encoding_ms': float,
        'retrieval_ms': float,
        'reranking_ms': float,
        'total_ms': float,
        'candidates_count': int,
        'final_count': int
    }
}
```

---

## Error Handling

All components raise standard Python exceptions:

```python
try:
    result = system.query(question)
except ConnectionError:
    print("Cannot connect to Milvus")
except ValueError:
    print("Invalid configuration")
except Exception as e:
    print(f"Unexpected error: {e}")
```
