import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.clients.qdrant_client import qdrant_client
from backend.config import config

def test_collection():
    """
    Test that the Qdrant collection exists and is properly configured
    """
    print("Testing Qdrant collection...")

    try:
        # Get collection info
        collection_info = qdrant_client.get_collection_info()
        if collection_info:
            print(f"✓ Collection {config.COLLECTION_NAME} exists")
            print(f"  - Points count: {collection_info.points_count}")
            print(f"  - Configured size: {collection_info.config.params.vectors_config.size}")
            print(f"  - Distance: {collection_info.config.params.vectors_config.distance}")
        else:
            print(f"✗ Failed to get collection info for {config.COLLECTION_NAME}")
            return False

        return True
    except Exception as e:
        print(f"✗ Error testing collection: {e}")
        return False

if __name__ == "__main__":
    test_collection()