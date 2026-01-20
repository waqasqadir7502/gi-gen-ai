#!/usr/bin/env python3
"""
Script to test the retrieval service functionality
"""
from backend.services.retrieval_service import retrieval_service

def test_retrieval():
    print("Testing retrieval service...")

    # Test a query similar to what the chatbot would receive
    test_query = "What is physical AI?"

    try:
        results = retrieval_service.retrieve_and_rerank(
            query=test_query,
            top_k=3
        )

        print(f"Retrieved {len(results)} chunks for query: '{test_query}'")

        if results:
            for i, result in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"  Score: {result.get('score', 'N/A')}")
                print(f"  Content preview: {result['content'][:100]}...")
                print(f"  Metadata keys: {list(result.get('metadata', {}).keys())}")
        else:
            print("No results returned from retrieval service!")

    except Exception as e:
        print(f"Error during retrieval: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_retrieval()