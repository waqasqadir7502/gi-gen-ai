#!/usr/bin/env python3
"""
Script to check if the Qdrant collection has been populated
"""
import os
from backend.config import config
from qdrant_client import QdrantClient

def check_qdrant_collection():
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
    )

    collection_name = "physical-ai-book-v1"

    # Get collection info
    collection_info = client.get_collection(collection_name)

    print(f"Collection: {collection_name}")
    print(f"Points count: {collection_info.points_count}")
    print(f"Indexed vectors: {collection_info.indexed_vectors_count}")

    if collection_info.points_count > 0:
        print("\nCollection has been populated successfully!")
        print(f"The chatbot should now be able to retrieve information from {collection_info.points_count} indexed points.")
    else:
        print("\nCollection is empty. The indexing may have failed.")

    return collection_info.points_count > 0

if __name__ == "__main__":
    check_qdrant_collection()