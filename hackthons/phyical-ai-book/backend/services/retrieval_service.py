from typing import List, Dict, Any

# Handle relative imports for direct execution
try:
    from ..clients.qdrant_client import QdrantRAGClient
    from ..clients.cohere_client import CohereClient
    from ..config import config
    from ..utils.logger import log_info, log_error, log_warning
    from ..utils.metadata_extractor import metadata_extractor
    from ..utils.db_connection import db_manager
    from ..models.document_model import Chunk
except (ImportError, ValueError):
    # Fallback for direct execution
    import sys
    from pathlib import Path
    # Add the backend directory to the path
    backend_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(backend_dir))

    from clients.qdrant_client import QdrantRAGClient
    from clients.cohere_client import CohereClient
    from config import config
    from utils.logger import log_info, log_error, log_warning
    from utils.metadata_extractor import metadata_extractor
    from utils.db_connection import db_manager
    from models.document_model import Chunk

# Create client instances
try:
    qdrant_client = QdrantRAGClient()
    cohere_client = CohereClient()
except Exception as e:
    print(f"Error initializing clients: {e}")
    # Create mock clients that return error responses
    class MockQdrantClient:
        def search(self, query_vector, top_k=5, filters=None):
            print("Mock Qdrant client called - client not available")
            return []

    class MockCohereClient:
        def embed_query(self, query):
            print("Mock Cohere client called - client not available")
            return [0.0] * 1024  # Return a dummy embedding

    qdrant_client = MockQdrantClient()
    cohere_client = MockCohereClient()

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

class RetrievalService:
    def __init__(self):
        self.top_k = config.TOP_K
        self.collection_name = config.COLLECTION_NAME

    def retrieve_relevant_chunks(self, query: str, top_k: int = None, filters: Dict[str, Any] = None) -> List[Dict]:
        """
        Retrieve relevant chunks based on the query using vector similarity search
        """
        if top_k is None:
            top_k = self.top_k

        # Check if Qdrant client is available
        if qdrant_client.client is None:
            log_warning("Qdrant client not available, returning empty results", extra={
                "query": query[:100] + "..." if len(query) > 100 else query
            })
            return []

        try:
            # Embed the query using Cohere
            query_embedding = cohere_client.embed_query(query)

            # Search in Qdrant for similar vectors
            search_results = qdrant_client.search(
                query_vector=query_embedding,
                top_k=top_k,
                filters=filters
            )

            # Process results and extract relevant information
            retrieved_chunks = []
            for result in search_results:
                chunk_data = {
                    'id': result.id,
                    'content': result.payload.get('content', ''),
                    'score': result.score,
                    'metadata': {
                        'document_id': result.payload.get('document_id', ''),
                        'file_path': result.payload.get('file_path', ''),
                        'relative_path': result.payload.get('relative_path', ''),
                        'source_title': result.payload.get('source_title', ''),
                        'source_heading': result.payload.get('source_heading', ''),
                        'content_type': result.payload.get('content_type', 'text'),
                        'chunk_index': result.payload.get('chunk_index', 0),
                        'total_chunks': result.payload.get('total_chunks', 1),
                        'token_count': result.payload.get('token_count', 0),
                        'content_length': result.payload.get('content_length', 0),
                        'indexed_at': result.payload.get('indexed_at', ''),
                    }
                }

                # Enrich with metadata from Neon database if available
                enriched_chunk = self._enrich_with_db_metadata(chunk_data, result.id)
                retrieved_chunks.append(enriched_chunk)

            log_info(f"Retrieved {len(retrieved_chunks)} chunks for query", extra={
                "query_length": len(query),
                "top_k": top_k,
                "actual_retrieved": len(retrieved_chunks)
            })

            return retrieved_chunks

        except Exception as e:
            log_error(f"Error during retrieval: {str(e)}", extra={
                "query": query[:100] + "..." if len(query) > 100 else query
            })
            return []

    def retrieve_with_context(self, query: str, context: str = None, top_k: int = None) -> List[Dict]:
        """
        Retrieve relevant chunks, optionally incorporating additional context
        """
        # If context is provided, combine it with the query for better retrieval
        if context:
            search_query = f"{query} {context}"
        else:
            search_query = query

        return self.retrieve_relevant_chunks(search_query, top_k)

    def retrieve_by_document_id(self, document_id: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve all chunks from a specific document
        """
        if top_k is None:
            top_k = self.top_k

        filters = {"document_id": document_id}

        return self.retrieve_relevant_chunks("retrieval query", top_k=top_k, filters=filters)

    def retrieve_by_file_path(self, file_path: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve chunks from a specific file
        """
        if top_k is None:
            top_k = self.top_k

        filters = {"file_path": file_path}

        return self.retrieve_relevant_chunks("retrieval query", top_k=top_k, filters=filters)

    def rerank_results(self, query: str, chunks: List[Dict], top_k: int = None) -> List[Dict]:
        """
        Rerank retrieved results for improved relevance
        """
        if not chunks:
            return chunks

        if top_k is None:
            top_k = len(chunks)

        try:
            # Extract content from chunks for reranking
            texts = [chunk['content'] for chunk in chunks]

            # Use Cohere's rerank functionality
            response = cohere_client.client.rerank(
                query=query,
                documents=texts,
                top_n=top_k
            )

            # Reorder the chunks based on reranking
            reranked_chunks = []
            for idx, result in enumerate(response.results):
                original_chunk = chunks[result.index]
                original_chunk['rerank_score'] = result.relevance_score
                original_chunk['rerank_position'] = idx + 1
                reranked_chunks.append(original_chunk)

            log_info(f"Reranked {len(chunks)} chunks to {len(reranked_chunks)} results", extra={
                "original_count": len(chunks),
                "reranked_count": len(reranked_chunks)
            })

            return reranked_chunks

        except Exception as e:
            log_error(f"Error during reranking: {str(e)}", extra={
                "query": query[:100] + "..." if len(query) > 100 else query,
                "chunk_count": len(chunks)
            })
            # If reranking fails, return original chunks with scores
            for i, chunk in enumerate(chunks):
                chunk['rerank_score'] = chunk.get('score', 0.0)
                chunk['rerank_position'] = i + 1
            return chunks[:top_k] if top_k else chunks

    def _enrich_with_db_metadata(self, chunk_data: Dict, qdrant_id: str) -> Dict:
        """
        Enrich chunk data with additional metadata from Neon database
        """
        if not db_manager.engine:
            # If database is not configured, return original data
            return chunk_data

        try:
            session = db_manager.get_session()

            # Look up chunk metadata from database using the Qdrant ID
            chunk_record = session.query(Chunk).filter(Chunk.chunk_id == qdrant_id).first()

            if chunk_record:
                # Add database metadata to the chunk
                db_metadata = {
                    'db_chunk_id': chunk_record.id,
                    'content_preview': chunk_record.content_preview,
                    'db_content_length': chunk_record.content_length,
                    'db_token_count': chunk_record.token_count,
                    'db_created_at': str(chunk_record.created_at),
                    'db_additional_metadata': chunk_record.metadata_json
                }

                # Merge with existing metadata
                chunk_data['metadata'].update(db_metadata)

            return chunk_data

        except SQLAlchemyError as e:
            log_error(f"Database error enriching metadata: {str(e)}")
            return chunk_data
        except Exception as e:
            log_error(f"Error enriching metadata from Neon database: {str(e)}")
            return chunk_data
        finally:
            if 'session' in locals():
                session.close()

    def retrieve_and_rerank(self, query: str, context: str = None, top_k: int = None) -> List[Dict]:
        """
        Retrieve and rerank results for maximum relevance
        """
        # First retrieve with context
        chunks = self.retrieve_with_context(query, context, top_k)

        # Then rerank the results
        reranked_chunks = self.rerank_results(query, chunks, top_k)

        return reranked_chunks

# Create a singleton instance
retrieval_service = RetrievalService()