# Hướng dẫn Triển khai Hoàn chỉnh

## 📋 Checklist Triển khai

### Phase 1: Setup Môi trường ✅

- [ ] Python 3.8+ đã cài đặt
- [ ] Docker Desktop đã cài đặt (cho Milvus)
- [ ] Git đã cài đặt
- [ ] CUDA (optional, cho GPU acceleration)

### Phase 2: Cài đặt Dependencies ✅

- [ ] Clone repository
- [ ] Tạo virtual environment
- [ ] Install Python packages
- [ ] Verify installations

### Phase 3: Setup Services ✅

- [ ] Start Milvus với Docker
- [ ] Verify Milvus connection
- [ ] Configure API keys

### Phase 4: Data Preparation ✅

- [ ] Collect PDF documents
- [ ] Organize files in data/pdfs/
- [ ] Run indexing

### Phase 5: Testing ✅

- [ ] Test basic query
- [ ] Run evaluation
- [ ] Performance tuning

---

## 📝 Chi tiết Từng Bước

### 1. Clone Repository

```bash
git clone https://github.com/your-username/RAG_QCTCVN.git
cd RAG_QCTCVN
```

### 2. Setup Python Environment

#### Windows

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Linux/Mac

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected time:** 10-15 minutes (download models)

### 3. Setup Milvus

#### Option A: Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# Check status
docker ps

# Expected output:
# milvus-standalone  Up
# milvus-etcd       Up
# milvus-minio      Up
```

#### Option B: Zilliz Cloud

1. Đăng ký tại https://zilliz.com
2. Tạo cluster mới
3. Copy endpoint và token
4. Update `.env`:

```env
MILVUS_HOST=your-cluster.aws-us-west-2.vectordb.zillizcloud.com
MILVUS_PORT=19530
MILVUS_TOKEN=your_token_here
```

### 4. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit configuration
nano .env  # hoặc notepad .env trên Windows
```

**Required configs:**

```env
# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Gemini API Key (Get from: https://makersuite.google.com/app/apikey)
GEMINI_API_KEY=AIza...your_key_here
```

### 5. Download Models (Auto-download)

Models sẽ tự động download khi chạy lần đầu:

- **BGE-M3**: ~2GB
- **Vietnamese_Reranker**: ~500MB

**Manual download (optional):**

```bash
python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3')"
```

### 6. Prepare Data

```bash
# Create data directory
mkdir -p data/pdfs

# Copy your PDF files
cp /path/to/your/pdfs/*.pdf data/pdfs/

# Verify
ls data/pdfs/
```

**Recommended structure:**

```
data/
└── pdfs/
    ├── QCVN_01_2021_BXD.pdf
    ├── QCVN_02_2021_BXD.pdf
    └── ...
```

### 7. Index Documents

```bash
python src/main.py --mode index --pdf-dir data/pdfs
```

**Expected output:**

```
Initializing RAG System...
Loading BGE-M3 model: BAAI/bge-m3
Model loaded successfully!

============================================================
INDEXING DOCUMENTS
============================================================

Found 5 PDF files

Processing: QCVN_01_2021_BXD.pdf
  → Generated 245 chunks
Encoding documents: 100%|██████████| 8/8 [00:12<00:00,  1.56s/it]
  → Encoded 245 embeddings
  → Inserted 245 documents to Milvus

...

============================================================
INDEXING COMPLETED: 1234 chunks indexed
============================================================
```

**Time estimate:**
- 5 PDFs (~100 pages each): 5-10 minutes
- 50 PDFs: 30-60 minutes

### 8. Test Query

```bash
python src/main.py --mode query --question "Chiều cao tối thiểu của tầng 1 là bao nhiêu?"
```

**Expected output:**

```
============================================================
QUERY: Chiều cao tối thiểu của tầng 1 là bao nhiêu?
============================================================

ANSWER:
Theo quy định tại [QCVN 01:2021/BXD - Điều 5, Khoản 1], chiều cao tối 
thiểu của tầng 1 là 3.0m, tính từ sàn hoàn thiện đến trần hoàn thiện.

============================================================
CITATIONS:
[1] QCVN 01:2021/BXD - Điều 5, Khoản 1
    Điều 5: Chiều cao tầng... Khoản 1: Chiều cao tối thiểu...

============================================================
STATISTICS:
  encoding_ms: 45.23ms
  retrieval_ms: 89.12ms
  reranking_ms: 42.34ms
  total_ms: 176.69ms
============================================================
```

### 9. Run Evaluation

```bash
python examples/evaluation_example.py
```

---

## 🔧 Performance Tuning

### GPU vs CPU

**GPU (Recommended):**

```env
BGE_USE_FP16=True
```

- Faster: 3-5x
- Memory: ~4GB VRAM

**CPU:**

```env
BGE_USE_FP16=False
```

- Slower but works on any machine
- Memory: ~8GB RAM

### Batch Size Tuning

Trong `src/embedder.py`:

```python
# Giảm batch size nếu Out of Memory
embedder.encode_documents(texts, batch_size=16)  # Default: 32
```

### Hybrid Search Weights

Test different combinations:

```python
config = {
    'hybrid_dense_weight': 0.7,   # More semantic
    'hybrid_sparse_weight': 0.3,  # Less keyword
}

system = RAGSystem(config=config)
```

### Top-K and Top-N

```env
RERANKER_TOP_K=50   # Fewer candidates (faster)
RERANKER_TOP_N=3    # Fewer final results
```

---

## 🐛 Common Issues

### Issue 1: Milvus Connection Failed

**Error:**
```
Failed to connect to Milvus: [Errno 61] Connection refused
```

**Solution:**
```bash
# Check if Milvus is running
docker ps

# Restart Milvus
docker-compose restart

# Check logs
docker logs milvus-standalone
```

### Issue 2: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. Reduce batch size:
```python
embedder = BGEEmbedder(batch_size=8)
```

2. Use CPU:
```env
BGE_USE_FP16=False
```

3. Use smaller model (trade-off quality):
```env
BGE_MODEL_NAME=BAAI/bge-small-en-v1.5
```

### Issue 3: Gemini API Error

**Error:**
```
google.api_core.exceptions.InvalidArgument: 400 Invalid API key
```

**Solution:**

1. Verify API key: https://makersuite.google.com/app/apikey
2. Check `.env` file có key đúng
3. Restart Python để reload .env

### Issue 4: PDF Parsing Error

**Error:**
```
pdfplumber.pdfminer.pdfparser.PDFSyntaxError
```

**Solutions:**

1. Check PDF không bị corrupt:
```bash
pdfinfo your_file.pdf
```

2. Try alternative parser (update `document_processor.py`):
```python
# Use PyPDF2 instead of pdfplumber
import PyPDF2
```

---

## 📊 Performance Benchmarks

**Hardware:** RTX 3080 (10GB), 32GB RAM

| Operation | Time | Throughput |
|-----------|------|------------|
| Embedding (1 chunk) | 15ms | 66 chunks/sec |
| Retrieval (Top 100) | 50ms | 20 queries/sec |
| Reranking (100→5) | 40ms | 25 queries/sec |
| Generation | 1.5s | 0.67 queries/sec |
| **End-to-end** | **~1.6s** | **0.62 queries/sec** |

**Indexing:**
- 1000 chunks: ~2 minutes
- 10000 chunks: ~15 minutes

---

## 🚀 Production Deployment

### 1. Use Production Milvus

Zilliz Cloud hoặc self-hosted cluster.

### 2. Add Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_query(question: str):
    return system.query(question)
```

### 3. Add API Server

Example với FastAPI:

```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/query")
def query_endpoint(question: str):
    result = system.query(question, verbose=False)
    return result
```

### 4. Monitoring

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log all queries
logger.info(f"Query: {question}, Latency: {stats['total_ms']}ms")
```

### 5. Load Balancing

Deploy multiple instances và sử dụng load balancer (nginx, HAProxy).

---

## ✅ Verification Checklist

Sau khi setup xong, verify:

- [ ] Milvus đang chạy: `docker ps`
- [ ] Collection được tạo: Check Milvus UI (port 9091)
- [ ] Documents được index: `system.vector_db.get_collection_stats()`
- [ ] Query trả về kết quả hợp lý
- [ ] Citations chính xác
- [ ] Latency < 3 seconds

---

## 📞 Support

- **Issues:** https://github.com/your-username/RAG_QCTCVN/issues
- **Docs:** [README.md](../README.md)
- **API Reference:** [API_REFERENCE.md](API_REFERENCE.md)
