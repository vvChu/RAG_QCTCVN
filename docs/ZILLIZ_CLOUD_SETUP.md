# Zilliz Cloud Setup Guide

## 📋 Tổng quan

Zilliz Cloud là managed service của Milvus, cung cấp vector database cloud-native với:
- ✅ Không cần Docker/self-hosting
- ✅ Auto-scaling và high availability
- ✅ Free tier: 1 cluster, 1GB storage
- ✅ TLS/SSL security

## 🔐 Thông tin kết nối của bạn

Từ file `zilliz-cloud-Free-01-username-password.txt`:

```
User: db_74b5693bc1c4c80
Password: Tg8+UKg4{{)ze9.(
```

## 🚀 Cấu hình Hệ thống

### Bước 1: Lấy Cluster Endpoint

1. Đăng nhập Zilliz Cloud: https://cloud.zilliz.com/
2. Vào Dashboard → Clusters
3. Copy **Cluster Endpoint** (VD: `in01-xxxx.aws-us-west-2.vectordb.zillizcloud.com`)

### Bước 2: Cập nhật `.env`

Tạo file `.env` từ `.env.example`:

```bash
cp .env.example .env
```

Sửa nội dung:

```env
# Milvus/Zilliz Cloud Configuration
MILVUS_HOST=your-cluster-endpoint.aws-region.vectordb.zillizcloud.com
MILVUS_PORT=19530
MILVUS_USER=db_74b5693bc1c4c80
MILVUS_PASSWORD=Tg8+UKg4{{)ze9.(
MILVUS_SECURE=True
MILVUS_COLLECTION_NAME=legal_documents

# BGE-M3 Configuration
BGE_MODEL_NAME=BAAI/bge-m3
BGE_MAX_LENGTH=8192
BGE_USE_FP16=True

# Reranker Configuration
RERANKER_MODEL_NAME=Vietnamese_Reranker
RERANKER_TOP_K=100
RERANKER_TOP_N=5

# Gemini CLI Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-pro

# Retrieval Configuration
HYBRID_DENSE_WEIGHT=0.5
HYBRID_SPARSE_WEIGHT=0.5
```

**⚠️ Quan trọng:**
- `MILVUS_HOST`: Thay bằng cluster endpoint thực tế
- `MILVUS_SECURE=True`: Bắt buộc cho Zilliz Cloud
- `MILVUS_USER` và `MILVUS_PASSWORD`: Credentials đã cung cấp

### Bước 3: Test kết nối

```bash
python -c "from src.vector_store import MilvusVectorDB; import os; from dotenv import load_dotenv; load_dotenv(); db = MilvusVectorDB(host=os.getenv('MILVUS_HOST'), port=os.getenv('MILVUS_PORT'), user=os.getenv('MILVUS_USER'), password=os.getenv('MILVUS_PASSWORD'), secure=True)"
```

**Expected output:**
```
✓ Connected to Zilliz Cloud at your-endpoint.vectordb.zillizcloud.com:19530
```

### Bước 4: Index và Query

Bây giờ có thể sử dụng bình thường:

```bash
# Index documents
python src/main.py --mode index --pdf-dir data/pdfs

# Query
python src/main.py --mode query --question "Chiều cao tối thiểu tầng 1?"
```

## 🔄 Chuyển đổi giữa Local và Cloud

### Sử dụng Local Milvus (Docker)

```env
# Local Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=
MILVUS_PASSWORD=
MILVUS_SECURE=False
```

```bash
docker-compose up -d
```

### Sử dụng Zilliz Cloud

```env
# Zilliz Cloud
MILVUS_HOST=your-endpoint.vectordb.zillizcloud.com
MILVUS_PORT=19530
MILVUS_USER=db_74b5693bc1c4c80
MILVUS_PASSWORD=Tg8+UKg4{{)ze9.(
MILVUS_SECURE=True
```

## 📊 Free Tier Limits

| Resource | Limit |
|----------|-------|
| Clusters | 1 |
| Storage | 1 GB |
| CU (Compute Units) | 1 CU |
| Collections | Unlimited |
| Vectors | ~1M (1024 dim) |

**Estimation cho Legal Documents:**
- 1000 chunks × 1024 dim × 4 bytes ≈ 4 MB
- **Free tier có thể lưu ~250,000 chunks**

## 🛠️ Troubleshooting

### Lỗi 1: Connection Timeout

```
MilvusException: Fail connecting to server
```

**Giải pháp:**
1. Verify endpoint chính xác
2. Check cluster status (phải "Running")
3. Verify username/password

### Lỗi 2: Authentication Failed

```
MilvusException: Authentication failed
```

**Giải pháp:**
1. Double-check `MILVUS_USER` và `MILVUS_PASSWORD`
2. Ensure `MILVUS_SECURE=True`
3. Copy credentials chính xác (không thêm spaces)

### Lỗi 3: Collection Already Exists

```
MilvusException: CreateCollection failed: collection already exists
```

**Giải pháp:**

```python
from pymilvus import utility, connections
utility.drop_collection("legal_documents")
```

Hoặc dùng parameter `drop_if_exists=True`:

```bash
python src/main.py --mode index --pdf-dir data/pdfs --force
```

## 🔒 Security Best Practices

### 1. Không commit `.env` vào Git

`.gitignore` đã có:
```
.env
```

### 2. Rotate credentials định kỳ

Trong Zilliz Cloud Dashboard → Security → Reset Password

### 3. Sử dụng environment-specific configs

```bash
# Development
cp .env.dev .env

# Production
cp .env.prod .env
```

## 📈 Monitoring

### Xem Usage trong Dashboard

1. Login Zilliz Cloud
2. Dashboard → Usage
3. Monitor:
   - Storage used
   - Query count
   - CU consumption

### Programmatic monitoring

```python
from src.main import RAGSystem

system = RAGSystem()
stats = system.vector_db.get_collection_stats()

print(f"Entities: {stats['num_entities']}")
print(f"Collection: {stats['name']}")
```

## 🚀 Migration từ Local sang Cloud

### 1. Export data từ Local Milvus

```python
# Chưa implement - TODO
# Cần export vectors và metadata từ local
```

### 2. Re-index trên Zilliz Cloud

Cách đơn giản nhất: Re-run indexing pipeline:

```bash
# Switch to Zilliz Cloud config
nano .env  # Update to Zilliz credentials

# Re-index
python src/main.py --mode index --pdf-dir data/pdfs
```

## 💡 Tips

### Tối ưu chi phí

- Xóa collections không dùng
- Monitor storage usage
- Upgrade khi cần thiết

### Performance

- Zilliz Cloud auto-scales
- Latency cao hơn local ~50-100ms (network overhead)
- Đổi lại là không cần quản lý infrastructure

### Backup

- Zilliz tự động backup
- Export data định kỳ (best practice)

## 📞 Support

- **Zilliz Docs**: https://docs.zilliz.com/
- **Dashboard**: https://cloud.zilliz.com/
- **Support**: support@zilliz.com

---

**✅ Bạn đã sẵn sàng sử dụng Zilliz Cloud!**

Next: [Index documents](../README.md#index-documents)
