"""
Example script: Indexing và Query đơn giản
"""

from src.main import RAGSystem

def main():
    # 1. Initialize RAG System
    print("Khởi tạo RAG System...")
    system = RAGSystem()
    
    # 2. Index documents (chỉ chạy 1 lần)
    print("\n=== INDEXING DOCUMENTS ===")
    system.index_documents(pdf_dir="data/pdfs")
    
    # 3. Test queries
    print("\n=== TESTING QUERIES ===")
    
    test_questions = [
        "Chiều cao tối thiểu của tầng 1 là bao nhiêu?",
        "Quy định về khoảng cách an toàn giữa các công trình?",
        "Yêu cầu về hệ thống phòng cháy chữa cháy?",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Question {i} ---")
        result = system.query(question)
        
        # Optional: Save results
        # import json
        # with open(f'results/query_{i}.json', 'w', encoding='utf-8') as f:
        #     json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
