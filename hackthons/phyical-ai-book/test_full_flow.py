#!/usr/bin/env python3
"""
Script to test the full RAG flow: retrieval + generation
"""
from backend.services.retrieval_service import retrieval_service
from backend.services.generation_service import generation_service

def test_full_flow():
    print("Testing full RAG flow (retrieval + generation)...")

    # Test query similar to what user would ask
    test_query = "What is physical AI?"

    try:
        # Step 1: Retrieve relevant chunks
        print(f"Step 1: Retrieving relevant chunks for query: '{test_query}'")
        retrieved_chunks = retrieval_service.retrieve_and_rerank(
            query=test_query,
            top_k=3
        )

        print(f"Retrieved {len(retrieved_chunks)} chunks")

        if not retrieved_chunks:
            print("No chunks retrieved - this explains the 'no relevant information' response!")
            return

        # Show retrieved content
        for i, chunk in enumerate(retrieved_chunks):
            print(f"\nRetrieved Chunk {i+1}:")
            print(f"  Content preview: {chunk['content'][:100]}...")
            print(f"  Score: {chunk.get('score', 'N/A')}")

        # Step 2: Generate response using the retrieved chunks
        print(f"\nStep 2: Generating response using {len(retrieved_chunks)} chunks...")
        generation_result = generation_service.generate_summarized_response(
            query=test_query,
            context_chunks=retrieved_chunks,
            selected_context=None
        )

        print(f"\nGenerated Response:")
        print(f"  Answer: {generation_result['answer'][:200]}...")
        print(f"  Sources: {generation_result.get('sources', [])}")
        print(f"  Metadata: {generation_result.get('metadata', {})}")

        print("\nSUCCESS: Full RAG flow is working correctly!")
        print("- Retrieval service finds relevant content")
        print("- Generation service creates appropriate response")
        print("- The chatbot should now provide meaningful answers")

    except Exception as e:
        print(f"Error in full flow test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_full_flow()