"""
Example: Evaluation với test dataset
"""

from src.main import RAGSystem
from src.evaluation import RAGEvaluator

def main():
    # Initialize
    system = RAGSystem()
    evaluator = RAGEvaluator()
    
    # Test dataset (question, ground_truth_doc_ids)
    test_cases = [
        {
            "question": "Chiều cao tối thiểu tầng 1?",
            "ground_truth": ["QCVN_01_2021_article_5"]
        },
        {
            "question": "Khoảng cách an toàn công trình?",
            "ground_truth": ["QCVN_01_2021_article_6"]
        },
    ]
    
    # Run evaluation
    results = []
    
    for i, case in enumerate(test_cases):
        print(f"\nEvaluating case {i+1}/{len(test_cases)}...")
        
        # Query
        result = system.query(case['question'], verbose=False)
        
        # Evaluate
        eval_result = evaluator.evaluate_query(
            query_result=result,
            ground_truth_docs=case['ground_truth']
        )
        
        results.append({
            'question': case['question'],
            'result': result,
            'evaluation': eval_result
        })
        
        # Print metrics
        print(f"  nDCG@10: {eval_result['retrieval_metrics'].get('nDCG@10', 0):.3f}")
        print(f"  MRR@10: {eval_result['retrieval_metrics'].get('MRR@10', 0):.3f}")
    
    # Batch evaluation
    batch_eval = evaluator.evaluate_batch(
        query_results=[r['result'] for r in results],
        ground_truths=[case['ground_truth'] for case in test_cases]
    )
    
    print("\n" + "="*60)
    print("BATCH EVALUATION RESULTS")
    print("="*60)
    print(f"Average nDCG@10: {batch_eval['retrieval_metrics'].get('nDCG@10', 0):.3f}")
    print(f"Average MRR@10: {batch_eval['retrieval_metrics'].get('MRR@10', 0):.3f}")
    print(f"Average Latency: {batch_eval['avg_latency_ms']:.2f}ms")


if __name__ == "__main__":
    main()
