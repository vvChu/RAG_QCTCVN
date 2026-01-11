import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
from typing import List
from ccba_rag.retrieval.vectorstores.milvus import MilvusStore
from ccba_rag.retrieval.embedder import BGEEmbedder as Embedder
from ccba_rag.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

def test_retrieval():
    # Initialize RAG components
    print("Initializing retrieval system...")
    store = MilvusStore()
    embedder = Embedder()
    store.load_collection()
    
    # Define test cases for the "fixed" documents
    test_cases = [
        {
            "query": "Thuyết minh quy hoạch xây dựng?",
            "expected_doc": "QCVN_01_2021_KIENTRUC_Quy_Hoach_Xay_Dung.pdf",
            "desc": "Check QCVN 01 retrieval (Table extraction)"
        },
        {
            "query": "Diện tích tối thiểu của phòng ở trong nhà ở?",
            "expected_doc": "TCVN_4451_2012_KIENTRUC_Nha_O_Nguyen_Tac_Thiet_Ke.pdf",
            "desc": "Check TCVN 4451 retrieval"
        },
        {
            "query": "Thông tin về đấu thầu phải đăng tải ở đâu?",
            "expected_doc": "TT 79_2025_TT-BTC hướng dẫn việc cung cấp, đăng tải thông tin về đấu thầu và mẫu hồ sơ đấu thầu trên Hệ thống mạng đấu thầu quốc gia do Bộ trưởng Bộ Tài chính ban hành.pdf",
            "desc": "Check TT 79 retrieval"
        }
    ]
    
    print("\nRunning Retrieval Tests...")
    print("=" * 60)
    
    passed = 0
    for case in test_cases:
        query = case["query"]
        expected = case["expected_doc"]
        print(f"\nTarget: {case['desc']}")
        print(f"Query:  {query}")
        
        # 1. Embed query
        embeddings = embedder.encode_queries([query])
        dense_vec = embeddings['dense'][0]
        sparse_vec = embeddings['sparse'][0]
        
        if hasattr(dense_vec, 'tolist'):
            dense_vec = dense_vec.tolist()
            
        print(f"Vector type: {type(dense_vec)}, Length: {len(dense_vec)}")
        
        # 2. Search
        results = store.search(
            query_dense=dense_vec,
            query_sparse=sparse_vec,
            top_k=5
        )
        
        # 3. Verify
        found = False
        print("Top 3 Results:")
        for i, res in enumerate(results[:3]):
            doc_name = res.chunk.document_name
            score = res.score
            # Snippet (first 100 chars)
            snippet = res.chunk.text[:100].replace('\n', ' ')
            
            print(f"  {i+1}. [{score:.4f}] {doc_name}")
            print(f"     Snippet: {snippet}...")
            
            if doc_name == expected:
                found = True
        
        if found:
            print("✅ TEST PASSED: Expected document retrieved.")
            passed += 1
        else:
            print(f"❌ TEST FAILED: Expected '{expected}' not in top 3.")
            
    print("\n" + "=" * 60)
    print(f"Final Score: {passed}/{len(test_cases)}")
    
    if passed == len(test_cases):
        print("🌟 ALL TESTS PASSED! Indexing is valid.")
    else:
        print("⚠️ SOME TESTS FAILED. Quality check needed.")

if __name__ == "__main__":
    test_retrieval()
