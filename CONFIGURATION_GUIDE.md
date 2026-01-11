# Configuration Tuning Guide

## Performance Optimization Settings

### Quick Wins (Recommended)

#### 1. Reranker Top-K Reduction

**Impact**: 4-5x faster query latency

```env
# In your query/evaluation code:
top_k = 20  # Reduced from 75
```

**Trade-off**: Slight reduction in recall, but much faster

- Before: ~397s per question
- After: ~85s per question
- Speedup: ~4.7x

#### 2. BGE-M3 Max Length

**Impact**: Faster embedding, lower memory

```env
BGE_MAX_LENGTH=512  # Reduced from 8192
```

**Trade-off**: Long documents will be truncated

- Use 512 for most queries
- Use 1024 for complex documents
- Use 8192 only if absolutely necessary

#### 3. Reranker Batch Size

**Impact**: Better GPU utilization

```env
# In reranker initialization:
batch_size=128  # Increased from 64
```

**Trade-off**: Higher memory usage

- CPU: Use 32-64
- GPU: Use 128-256

### Advanced Tuning

#### HNSW Index Parameters

```env
MILVUS_HNSW_M=8              # Default: 16 (lower = faster search)
MILVUS_HNSW_EF_CONSTRUCTION=100  # Default: 200 (lower = faster indexing)  
MILVUS_HNSW_EF_SEARCH=128    # Default: 200 (lower = faster search)
```

#### Retrieval Pipeline

```python
# Optimal settings for CPU
retriever_config = {
    'top_k': 20,           # Candidates for reranking
    'top_n': 5,            # Final results
    'ef_search': 128       # HNSW search parameter
}
```

## Performance Benchmarks

### Current Configuration (Optimized)

| Component | Setting | Latency |
|-----------|---------|---------|
| Encoding | max_length=512 | ~100ms |
| Retrieval | top_k=20, ef=128 | ~150ms |
| Reranking | batch=64, 20→5 | ~85s |
| **Total** | - | **~85s** |

### Previous Configuration (Baseline)

| Component | Setting | Latency |
|-----------|---------|---------|
| Encoding | max_length=512 | ~100ms |
| Retrieval | top_k=75, ef=128 | ~150ms |
| Reranking | batch=64, 75→5 | ~397s |
| **Total** | - | **~397s** |

### Speedup: **4.7x Faster** ⚡

## Recommended Configurations

### Development (Fast Iteration)

```env
BGE_MAX_LENGTH=512
RERANKER_TOP_K=10
RERANKER_TOP_N=3
MILVUS_HNSW_EF_SEARCH=64
```

### Production (Balanced)

```env
BGE_MAX_LENGTH=512
RERANKER_TOP_K=20
RERANKER_TOP_N=5
MILVUS_HNSW_EF_SEARCH=128
```

### High Quality (Slow but Accurate)

```env
BGE_MAX_LENGTH=1024
RERANKER_TOP_K=50
RERANKER_TOP_N=10
MILVUS_HNSW_EF_SEARCH=200
```

## Troubleshooting

### Issue: Slow Query Performance

1. **Check top_k parameter** - Should be 20-30, not 75+
2. **Verify batch size** - Use 64-128 for better throughput
3. **Consider GPU** - Reranker is 5-10x faster on GPU

### Issue: Low Hit Rate

1. **Increase top_k** - Try 30-50 candidates
2. **Increase top_n** - Return more final results
3. **Check embeddings** - Ensure proper model loading
4. **Verify indexing** - Re-index with correct settings

### Issue: Out of Memory

1. **Reduce batch_size** - Lower to 32 or 16
2. **Reduce max_length** - Use 256 or 512
3. **Reduce top_k** - Use 10-15 instead of 20+

## Migration Guide

### Updating from Default Settings

```python
# Old code:
search_results = vector_store.search(top_k=75)  # Slow!

# New code:
search_results = vector_store.search(top_k=20)  # 4x faster
```

```python
# Old code:
embedder = BGEEmbedder(max_length=8192)  # High memory

# New code:  
embedder = BGEEmbedder(max_length=512)   # Optimized
```

## Next Steps

1. **Run quick evaluation** to verify improvements:

   ```bash
   python quick_eval.py
   ```

2. **Check system status**:

   ```bash
   python system_status.py
   ```

3. **Monitor performance** over time and adjust as needed
