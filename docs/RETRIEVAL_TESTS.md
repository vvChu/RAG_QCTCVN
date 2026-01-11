# Standardized Retrieval Tests

This document describes the standardized procedure for validating hybrid (dense + sparse) retrieval, reranking, and grounded generation in the Vietnamese Legal RAG system.

## Objectives

- Ensure hybrid fusion (dense + sparse) works and exposes `hybrid_score`, `dense_score`, `sparse_score`.
- Measure latency of each stage: embedding, hybrid search, reranking, generation.
- Produce reproducible test artifacts for benchmarking and regression detection.

## Script

`examples/sample_retrieval_tests.py` generates a JSON report `examples/sample_retrieval_report.json`.

## Queries

1. Khoảng cách phòng cháy giữa 2 tòa nhà là bao nhiêu?
2. Chiều cao tối thiểu của tầng 1 là bao nhiêu?
3. Thủ tục xin giấy phép xây dựng gồm những gì?
4. Yêu cầu về an toàn cháy đối với nhà cao tầng?

## Usage

```bash
python examples/sample_retrieval_tests.py
```

## Output Structure (excerpt)

```json
{
  "query": "Khoảng cách phòng cháy giữa 2 tòa nhà là bao nhiêu?",
  "timings_ms": {
    "embedding_ms": 642.1,
    "hybrid_search_ms": 248.3,
    "rerank_ms": 128533.0,
    "generation_ms": 811.9
  },
  "top_k": [
    {
      "rank": 1,
      "hybrid_score": 0.8735,
      "dense_score": 0.6578,
      "sparse_score": 12.431,
      "rerank_score": 0.9421,
      "document_name": "175_2024_ND-CP_609382.pdf",
      "article": "7",
      "clause": "6",
      "snippet": "..."
    }
  ],
  "answer": "...",
  "citations": [
    {"document_name": "175_2024_ND-CP_609382.pdf", "article": "7", "clause": "6"}
  ],
  "model": "gemini-2.5-flash"
}
```

## Notes

- Reranking latency is high on CPU (cross-encoder). Consider quantization or reducing `TOP_K`.
- Hybrid fusion applies min-max normalization within each modality before weighting.
- Slight >100% coverage originates from split boundary duplication; acceptable until GPU production mode (8192 tokens) is enabled.

## Next Improvements

- Add nDCG@10, MRR@10 automatic evaluator.
- Persist latency metrics historically for trend analysis.
- Integrate faithfulness evaluation (LLM-as-Judge) after generation.
