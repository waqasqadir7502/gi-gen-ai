#!/usr/bin/env python3
"""
Script to test Qdrant search functionality
"""
from backend.clients.qdrant_client import qdrant_client
from backend.clients.cohere_client import cohere_client
from backend.config import config

def test_search():
    print("Testing Qdrant search functionality...")

    # Test if collection exists and has points
    try:
        info = qdrant_client.get_collection_info()
        print(f"Collection info: {info}")
        print(f"Points count: {info.points_count}")
    except Exception as e:
        print(f"Error getting collection info: {e}")
        return

    if info.points_count == 0:
        print("Collection is empty. Please run ingestion first.")
        return

    # Test embedding a simple query
    test_query = "What is physical AI?"
    try:
        query_embedding = cohere_client.embed_query(test_query)
        print(f"Query embedding generated, length: {len(query_embedding)}")
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return

    # Test search
    try:
        search_results = qdrant_client.search(
            query_vector=query_embedding,
            top_k=3
        )
        print(f"Search completed, found {len(search_results)} results")

        if search_results:
            for i, result in enumerate(search_results):
                print(f"Result {i+1}: Score={result.score}, Content='{result.payload.get('content', '')[:100]}...'")
        else:
            print("No results found - this indicates the search is not working properly")

    except Exception as e:
        print(f"Error during search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_search()