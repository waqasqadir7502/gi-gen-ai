#!/usr/bin/env python3
"""
Script to reset the Qdrant collection for fresh indexing
"""
import os
from backend.config import config
from qdrant_client import QdrantClient

def reset_qdrant_collection():
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
    )

    collection_name = "physical-ai-book-v1"

    print(f"Dropping collection: {collection_name}")
    client.delete_collection(collection_name)
    print(f"Collection {collection_name} dropped.")

    print(f"Creating collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config={"size": 1024, "distance": "Cosine"}
    )
    print(f"Collection {collection_name} recreated.")

    print("Qdrant collection reset complete!")

if __name__ == "__main__":
    reset_qdrant_collection()