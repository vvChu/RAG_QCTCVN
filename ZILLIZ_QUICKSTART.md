# 🚀 Quick Connect to Zilliz Cloud

## Copy này vào `.env` của bạn:

```env
# Zilliz Cloud Free Tier
MILVUS_HOST=YOUR_CLUSTER_ENDPOINT_HERE.vectordb.zillizcloud.com
MILVUS_PORT=19530
MILVUS_USER=db_74b5693bc1c4c80
MILVUS_PASSWORD=Tg8+UKg4{{)ze9.(
MILVUS_SECURE=True
MILVUS_COLLECTION_NAME=legal_documents

# BGE-M3
BGE_MODEL_NAME=BAAI/bge-m3
BGE_MAX_LENGTH=8192
BGE_USE_FP16=True

# Reranker
RERANKER_MODEL_NAME=Vietnamese_Reranker
RERANKER_TOP_K=100
RERANKER_TOP_N=5

# Gemini API (Primary)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# Groq API (Fallback - Optional)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Hybrid Search
HYBRID_DENSE_WEIGHT=0.5
HYBRID_SPARSE_WEIGHT=0.5
```

## 📝 Bước thực hiện:

1. **Lấy Cluster Endpoint:**
   - Login: https://cloud.zilliz.com/
   - Dashboard → Clusters
   - Copy endpoint (VD: `in01-xxxx.aws-us-west-2.vectordb.zillizcloud.com`)

2. **Thay MILVUS_HOST:**
   ```
   MILVUS_HOST=<paste-endpoint-here>
   ```

3. **Test connection:**
   ```bash
   python src/main.py --mode query --question "test"
   ```

4. **Expected output:**
   ```
   ✓ Connected to Zilliz Cloud at your-endpoint.vectordb.zillizcloud.com:19530
   ```

## ✅ Bạn đã sẵn sàng!

Chi tiết: [docs/ZILLIZ_CLOUD_SETUP.md](docs/ZILLIZ_CLOUD_SETUP.md)
