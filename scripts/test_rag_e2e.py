"""
Comprehensive E2E Test for CCBA RAG System

Tests:
1. Package imports
2. Settings loading
3. Embedder initialization
4. Vector DB connection
5. Retrieval pipeline
6. Generation (if API keys available)
"""

import sys
import time
sys.path.insert(0, 'src')

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_result(test_name, passed, details=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} | {test_name}")
    if details:
        print(f"         └─ {details}")

def main():
    results = []
    
    # ========== TEST 1: Package Imports ==========
    print_section("TEST 1: Package Imports")
    
    try:
        from ccba_rag import RAGSystem
        print_result("Import RAGSystem", True)
        results.append(True)
    except Exception as e:
        print_result("Import RAGSystem", False, str(e))
        results.append(False)
        return  # Cannot continue without this
    
    try:
        from ccba_rag.core.settings import settings
        print_result("Import settings", True, f"Collection: {settings.milvus_collection_name}")
        results.append(True)
    except Exception as e:
        print_result("Import settings", False, str(e))
        results.append(False)
    
    try:
        from ccba_rag.core.prompts import prompt_manager
        print_result("Import prompt_manager", True, f"System instruction: {len(prompt_manager.system_instruction)} chars")
        results.append(True)
    except Exception as e:
        print_result("Import prompt_manager", False, str(e))
        results.append(False)
    
    try:
        from ccba_rag.generation.factory import create_generator
        print_result("Import generator factory", True)
        results.append(True)
    except Exception as e:
        print_result("Import generator factory", False, str(e))
        results.append(False)
    
    # ========== TEST 2: RAG System Initialization ==========
    print_section("TEST 2: RAG System Initialization")
    
    try:
        start = time.time()
        system = RAGSystem(verbose=False)
        init_time = time.time() - start
        print_result("Create RAGSystem", True, f"{init_time:.2f}s")
        results.append(True)
    except Exception as e:
        print_result("Create RAGSystem", False, str(e))
        results.append(False)
        return
    
    # ========== TEST 3: Embedder ==========
    print_section("TEST 3: Embedder")
    
    try:
        start = time.time()
        embedder = system.embedder
        load_time = time.time() - start
        dim = embedder.get_embedding_dim()
        print_result("Load BGE-M3 Embedder", True, f"dim={dim}, load_time={load_time:.2f}s")
        results.append(True)
    except Exception as e:
        print_result("Load BGE-M3 Embedder", False, str(e))
        results.append(False)
    
    try:
        test_text = "Chiều cao tối thiểu của tầng 1"
        start = time.time()
        dense, sparse = embedder.encode_all([test_text], show_progress=False)
        encode_time = (time.time() - start) * 1000
        print_result("Encode test query", True, f"dense_dim={len(dense[0])}, sparse_keys={len(sparse[0])}, time={encode_time:.0f}ms")
        results.append(True)
    except Exception as e:
        print_result("Encode test query", False, str(e))
        results.append(False)
    
    # ========== TEST 4: Vector DB ==========
    print_section("TEST 4: Vector DB (Milvus/Zilliz)")
    
    try:
        vector_db = system.vector_db
        has_collection = vector_db.has_collection()
        print_result("Connect to Milvus", True, f"collection_exists={has_collection}")
        results.append(True)
    except Exception as e:
        print_result("Connect to Milvus", False, str(e))
        results.append(False)
    
    # ========== TEST 5: Retrieval ==========
    print_section("TEST 5: Retrieval Pipeline")
    
    test_query = "Yêu cầu về phòng cháy chữa cháy trong tòa nhà cao tầng"
    
    try:
        start = time.time()
        result = system.retrieve(test_query, top_k=50, top_n=5)
        retrieval_time = (time.time() - start) * 1000
        
        contexts = result.get('contexts', [])
        if contexts:
            print_result("Retrieve contexts", True, f"found={len(contexts)}, time={retrieval_time:.0f}ms")
            print(f"\n  Top result:")
            print(f"    - Doc: {contexts[0].get('document_name', 'Unknown')}")
            print(f"    - Score: {contexts[0].get('rerank_score', contexts[0].get('retrieval_score', 0)):.4f}")
            print(f"    - Text: {contexts[0].get('text', '')[:100]}...")
            results.append(True)
        else:
            print_result("Retrieve contexts", False, "No contexts found - index may be empty")
            results.append(False)
    except Exception as e:
        print_result("Retrieve contexts", False, str(e))
        results.append(False)
    
    # ========== TEST 6: Full RAG Query ==========
    print_section("TEST 6: Full RAG Query (Retrieval + Generation)")
    
    try:
        start = time.time()
        result = system.query(
            question="Chiều cao tối thiểu của tầng 1 nhà ở là bao nhiêu?",
            verbose=False,
            top_k=30,
            top_n=3
        )
        total_time = (time.time() - start) * 1000
        
        answer = result.get('answer', '')
        model = result.get('model', 'Unknown')
        stats = result.get('stats', {})
        
        if answer and len(answer) > 50:
            print_result("Full RAG Query", True, f"model={model}, time={total_time:.0f}ms")
            print(f"\n  Answer preview:")
            print(f"    {answer[:200]}...")
            print(f"\n  Stats:")
            print(f"    - Retrieval: {stats.get('retrieval_ms', 0):.0f}ms")
            print(f"    - Generation: {stats.get('generation_ms', 0):.0f}ms")
            print(f"    - Contexts used: {stats.get('final_count', 0)}")
            results.append(True)
        else:
            print_result("Full RAG Query", False, f"Short or empty answer: {answer[:100]}")
            results.append(False)
            
    except Exception as e:
        print_result("Full RAG Query", False, str(e))
        results.append(False)
    
    # ========== SUMMARY ==========
    print_section("TEST SUMMARY")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n  Passed: {passed}/{total}")
    print(f"  Success Rate: {passed/total*100:.0f}%")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!")
    else:
        print(f"\n  ⚠️ {total - passed} test(s) failed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
