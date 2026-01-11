import os
import time
import json
from typing import List

from dotenv import load_dotenv
load_dotenv()

# Local imports (ensure src in path)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from embedder import BGEEmbedder  # type: ignore
from vector_store import MilvusVectorDB  # type: ignore
from reranker import VietnameseReranker  # type: ignore
from generator import GeminiGenerator  # type: ignore

DENSE_WEIGHT = float(os.getenv('HYBRID_DENSE_WEIGHT', '0.5'))
SPARSE_WEIGHT = float(os.getenv('HYBRID_SPARSE_WEIGHT', '0.5'))
TOP_K = int(os.getenv('RERANKER_TOP_K', '50'))
TOP_N = int(os.getenv('RERANKER_TOP_N', '5'))

QUERIES: List[str] = [
    "Khoảng cách phòng cháy giữa 2 tòa nhà là bao nhiêu?",
    "Chiều cao tối thiểu của tầng 1 là bao nhiêu?",
    "Thủ tục xin giấy phép xây dựng gồm những gì?",
    "Yêu cầu về an toàn cháy đối với nhà cao tầng?"
]

def run_query(query: str, embedder: BGEEmbedder, db: MilvusVectorDB, reranker: VietnameseReranker, generator: GeminiGenerator):
    timings = {}

    t0 = time.time()
    enc = embedder.encode_queries([query])
    timings['embedding_ms'] = (time.time() - t0) * 1000

    q_dense = enc['dense']
    q_sparse = enc['sparse'][0]

    t1 = time.time()
    raw_results = db.hybrid_search(
        query_dense=q_dense,
        query_sparse=q_sparse,
        top_k=TOP_K,
        dense_weight=DENSE_WEIGHT,
        sparse_weight=SPARSE_WEIGHT
    )
    timings['hybrid_search_ms'] = (time.time() - t1) * 1000

    # Prepare pairs for reranker
    pairs = [(query, r['text']) for r in raw_results]

    # Rerank all candidates (keep TOP_K limit)
    t2 = time.time()
    ranked_docs, rerank_stats = reranker.rerank_with_details(
        query=query,
        documents=raw_results,
        top_n=TOP_K
    )
    timings['rerank_ms'] = rerank_stats['latency_ms']

    final_contexts = ranked_docs[:TOP_N]

    t3 = time.time()
    answer_payload = generator.generate(
        query=query,
        contexts=final_contexts
    )
    timings['generation_ms'] = (time.time() - t3) * 1000

    # Structure output
    output = {
        'query': query,
        'timings_ms': timings,
        'top_k': [
            {
                'rank': i + 1,
                'hybrid_score': r.get('hybrid_score'),
                'dense_score': r.get('dense_score'),
                'sparse_score': r.get('sparse_score'),
                'rerank_score': r.get('rerank_score'),
                'document_name': r.get('document_name'),
                'article': r.get('article'),
                'clause': r.get('clause'),
                'snippet': (r.get('text') or '')[:220]
            }
            for i, r in enumerate(final_contexts)
        ],
        'answer': answer_payload.get('answer'),
        'citations': [
            {
                'document_name': c.get('document_name'),
                'article': c.get('article'),
                'clause': c.get('clause')
            } for c in final_contexts
        ],
        'model': answer_payload.get('model')
    }
    return output


def main():
    print("[INFO] Initializing components for standardized retrieval tests...")
    embedder = BGEEmbedder(max_length=int(os.getenv('BGE_MAX_LENGTH', '512')))
    db = MilvusVectorDB(
        host=os.getenv('MILVUS_HOST', 'localhost'),
        port=os.getenv('MILVUS_PORT', '19530'),
        user=os.getenv('MILVUS_USER', ''),
        password=os.getenv('MILVUS_PASSWORD', ''),
        secure=os.getenv('MILVUS_SECURE', 'False').lower() == 'true',
        collection_name=os.getenv('MILVUS_COLLECTION_NAME', 'legal_documents')
    )
    db.load_collection()

    reranker = VietnameseReranker(
        model_name=os.getenv('RERANKER_MODEL_NAME', 'BAAI/bge-reranker-v2-m3'),
        batch_size=int(os.getenv('RERANKER_BATCH_SIZE', '64'))
    )
    generator = GeminiGenerator(
        api_key=os.getenv('GEMINI_API_KEY'),
        model=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    )

    all_results = []
    for q in QUERIES:
        print(f"\n[QUERY] {q}")
        res = run_query(q, embedder, db, reranker, generator)
        print(json.dumps(res['timings_ms'], ensure_ascii=False))
        print("Answer:", res['answer'])
        print("Top contexts:")
        for c in res['top_k']:
            print(f"  #{c['rank']} HS={c['hybrid_score']:.4f} RS={c['rerank_score']:.4f} | {c['document_name']} Điều {c['article']} Khoản {c['clause']}")
        all_results.append(res)

    # Save consolidated report
    out_path = os.path.join(os.path.dirname(__file__), 'sample_retrieval_report.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Saved standardized retrieval report to {out_path}")


if __name__ == '__main__':
    main()
