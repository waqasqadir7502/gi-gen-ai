from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any
import uuid

# Handle relative imports for direct execution
try:
    from ..config import config
except (ImportError, ValueError):
    # Fallback for direct execution
    import sys
    from pathlib import Path
    # Add the backend directory to the path
    backend_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(backend_dir))
    from config import config

class QdrantRAGClient:
    def __init__(self):
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )

        # Try to create collection if it doesn't exist, but don't fail on connection issues
        try:
            self._create_collection_if_not_exists()
        except Exception as e:
            print(f"Warning: Could not connect to Qdrant during initialization: {e}")
            print("The application will continue to run but may have issues with vector storage.")
            print("Make sure your Qdrant credentials are correct.")

    def _create_collection_if_not_exists(self):
        """
        Create the collection if it doesn't exist
        """
        try:
            collections = self.client.get_collections()
            collection_names = [collection.name for collection in collections.collections]

            if config.COLLECTION_NAME not in collection_names:
                self.client.create_collection(
                    collection_name=config.COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=config.VECTOR_SIZE,
                        distance=models.Distance.COSINE
                    ),
                )
                print(f"Created collection: {config.COLLECTION_NAME}")
            else:
                print(f"Collection {config.COLLECTION_NAME} already exists")
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise

    def upsert_vectors(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: List[str] = None):
        """
        Upsert vectors into the collection
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

        try:
            self.client.upsert(
                collection_name=config.COLLECTION_NAME,
                points=models.Batch(
                    ids=ids,
                    vectors=vectors,
                    payloads=payloads
                )
            )
            return True
        except Exception as e:
            print(f"Error upserting vectors: {e}")
            return False

    def search(self, query_vector: List[float], top_k: int = 8, filters: Dict[str, Any] = None):
        """
        Search for similar vectors in the collection
        """
        try:
            search_filter = None
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    filter_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )

                if filter_conditions:
                    search_filter = models.Filter(must=filter_conditions)

            results = self.client.search(
                collection_name=config.COLLECTION_NAME,
                query_vector=query_vector,
                limit=top_k,
                query_filter=search_filter
            )

            return results
        except Exception as e:
            # Try the older parameter name if the new one fails
            try:
                results = self.client.search(
                    collection_name=config.COLLECTION_NAME,
                    vector=query_vector,
                    limit=top_k,
                    query_filter=search_filter
                )
                return results
            except Exception as e2:
                print(f"Error searching vectors: {e}, fallback error: {e2}")
                return []

    def delete_collection(self):
        """
        Delete the entire collection (use with caution!)
        """
        try:
            self.client.delete_collection(collection_name=config.COLLECTION_NAME)
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False

    def get_collection_info(self):
        """
        Get information about the collection
        """
        try:
            info = self.client.get_collection(collection_name=config.COLLECTION_NAME)
            return info
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return None

# Create a singleton instance
qdrant_client = QdrantRAGClient()