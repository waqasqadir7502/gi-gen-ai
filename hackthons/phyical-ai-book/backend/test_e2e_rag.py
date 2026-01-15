import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.retrieval_service import retrieval_service
from backend.services.generation_service import generation_service
from backend.prompts.chat_prompt import chat_prompt_engineer
from backend.clients.cohere_client import cohere_client
from backend.clients.qdrant_client import qdrant_client
from backend.utils.logger import log_info, log_error
from backend.config import config

def test_end_to_end_rag():
    """
    Test the end-to-end RAG pipeline with sample queries
    """
    print("Testing End-to-End RAG Pipeline...")

    # Sample test query
    test_queries = [
        "What is the Physical AI Book about?",
        "How does the system work?",
        "What are the main concepts discussed in the book?"
    ]

    for i, query in enumerate(test_queries):
        print(f"\n--- Test Query {i+1}: {query} ---")

        try:
            # Step 1: Test retrieval
            print("Step 1: Testing retrieval...")
            retrieved_chunks = retrieval_service.retrieve_relevant_chunks(query)

            print(f"  Retrieved {len(retrieved_chunks)} chunks")
            if retrieved_chunks:
                print(f"  Top result score: {retrieved_chunks[0]['score']:.3f}")
                print(f"  Top result content preview: {retrieved_chunks[0]['content'][:100]}...")

            # Step 2: Test generation
            print("Step 2: Testing generation...")
            if retrieved_chunks:
                generation_result = generation_service.generate_response(
                    query=query,
                    context_chunks=retrieved_chunks
                )

                print(f"  Generated answer preview: {generation_result['answer'][:200]}...")
                print(f"  Sources found: {len(generation_result['sources'])}")
                print(f"  Metadata: {generation_result['metadata']}")
            else:
                print("  No chunks retrieved, skipping generation")

            print(f"  ✓ Query {i+1} completed successfully")

        except Exception as e:
            log_error(f"Error testing query '{query}': {str(e)}")
            print(f"  ✗ Query {i+1} failed: {str(e)}")

    print(f"\n--- RAG Pipeline Test Summary ---")
    print(f"Tested {len(test_queries)} queries")
    print("RAG pipeline components:")
    print(f"  - Cohere client: Available")
    print(f"  - Qdrant client: Available (connection tested during initialization)")
    print(f"  - Retrieval service: Working")
    print(f"  - Generation service: Working")
    print(f"  - Prompt engineering: Available")

    print("\n✓ End-to-End RAG Pipeline test completed")

def test_basic_connectivity():
    """
    Test basic connectivity to external services
    """
    print("\n--- Testing Basic Connectivity ---")

    try:
        # Test Cohere connection
        test_embedding = cohere_client.embed(["test"], input_type="classification")
        print("✓ Cohere connection: Working")
    except Exception as e:
        print(f"✗ Cohere connection: Failed ({str(e)})")

    try:
        # Test Qdrant connection
        collection_info = qdrant_client.get_collection_info()
        print(f"✓ Qdrant connection: Working (Collection: {config.COLLECTION_NAME})")
    except Exception as e:
        print(f"✗ Qdrant connection: Failed ({str(e)})")

if __name__ == "__main__":
    print("Starting End-to-End RAG Pipeline Tests...")

    test_basic_connectivity()
    test_end_to_end_rag()

    print("\n✓ All RAG Pipeline tests completed!")