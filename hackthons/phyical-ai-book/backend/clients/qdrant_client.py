import requests
import json
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
        # Check if required config is available before initializing
        if not config.QDRANT_URL or not config.QDRANT_API_KEY:
            print("Warning: Qdrant configuration not found. Qdrant client will be in offline mode.")
            self.base_url = None
            self.api_key = None
            return

        self.base_url = config.QDRANT_URL.rstrip('/')
        self.api_key = config.QDRANT_API_KEY
        self.collection_name = config.COLLECTION_NAME

    def search(self, query_vector: List[float], top_k: int = 8, filters: Dict[str, Any] = None):
        """
        Search for similar vectors in the collection using HTTP requests
        """
        if not self.base_url or not self.api_key:
            print("Qdrant client not available, returning empty results")
            return []

        try:
            headers = {
                "api-key": self.api_key,
                "Content-Type": "application/json"
            }

            search_payload = {
                "vector": query_vector,
                "limit": top_k,
                "with_payload": True,
                "with_vectors": False
            }

            if filters:
                search_payload["filter"] = self._convert_filters(filters)

            response = requests.post(
                f"{self.base_url}/collections/{self.collection_name}/points/search",
                headers=headers,
                json=search_payload
            )

            if response.status_code == 200:
                results = response.json()
                return results.get("result", [])
            else:
                print(f"Qdrant search request failed: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"Error during Qdrant search: {e}")
            return []

    def upsert(self, vectors: List[List[float]], payloads: List[Dict], ids: List[str] = None):
        """
        Upsert vectors to the collection using HTTP requests
        """
        if not self.base_url or not self.api_key:
            print("Qdrant client not available, skipping upsert")
            return False

        try:
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

            points = []
            for i, (vector, payload) in enumerate(zip(vectors, payloads)):
                point_id = ids[i] if i < len(ids) else str(uuid.uuid4())
                points.append({
                    "id": point_id,
                    "vector": vector,
                    "payload": payload
                })

            headers = {
                "api-key": self.api_key,
                "Content-Type": "application/json"
            }

            upsert_payload = {
                "points": points
            }

            response = requests.put(
                f"{self.base_url}/collections/{self.collection_name}/points?wait=true",
                headers=headers,
                json=upsert_payload
            )

            if response.status_code in [200, 202]:
                return True
            else:
                print(f"Qdrant upsert request failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Error during Qdrant upsert: {e}")
            return False

    def _convert_filters(self, filters: Dict[str, Any]):
        """
        Convert simple filters to Qdrant filter format
        """
        if not filters:
            return {}

        must_conditions = []
        for key, value in filters.items():
            must_conditions.append({
                "key": key,
                "match": {"value": value}
            })

        return {"must": must_conditions}

    def health(self):
        """
        Check if Qdrant is accessible
        """
        if not self.base_url or not self.api_key:
            return False

        try:
            headers = {
                "api-key": self.api_key,
                "Content-Type": "application/json"
            }

            response = requests.get(f"{self.base_url}/collections/{self.collection_name}", headers=headers)

            return response.status_code == 200
        except Exception:
            return False

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