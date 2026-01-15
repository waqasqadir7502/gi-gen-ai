from typing import List, Dict, Any
from ..clients.cohere_client import cohere_client
from ..config import config
from ..utils.logger import log_info, log_error
from ..utils.chunking import chunker
from ..utils.metadata_extractor import metadata_extractor
import numpy as np

class EmbeddingPipeline:
    def __init__(self):
        # Use the embedding model from the Cohere client
        from ..clients.cohere_client import cohere_client
        self.model = cohere_client.embedding_model
        self.input_type = "search_document"  # Default input type for documents

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Embed a list of text chunks using Cohere's embedding model
        """
        if not chunks:
            return []

        # Extract just the content from chunks for embedding
        texts = [chunk['content'] for chunk in chunks]

        try:
            # Generate embeddings for all texts at once
            embeddings = cohere_client.embed(texts, input_type=self.input_type)

            # Create result list with embeddings and original chunk data
            result = []
            for i, chunk in enumerate(chunks):
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding['embedding'] = embeddings[i]
                result.append(chunk_with_embedding)

            log_info(f"Successfully embedded {len(chunks)} chunks", extra={
                "model": self.model,
                "input_type": self.input_type
            })

            return result

        except Exception as e:
            log_error(f"Error during embedding: {str(e)}")
            raise

    def embed_single_text(self, text: str, input_type: str = "search_document") -> List[float]:
        """
        Embed a single text using Cohere's embedding model
        """
        try:
            embeddings = cohere_client.embed([text], input_type=input_type)
            return embeddings[0]  # Return the first (and only) embedding
        except Exception as e:
            log_error(f"Error embedding single text: {str(e)}")
            raise

    def process_document_for_embedding(self, document: Dict) -> List[Dict]:
        """
        Process an entire document: chunk it, extract metadata, and embed
        """
        try:
            # Step 1: Chunk the document
            chunks = chunker.chunk_document(document)

            # Step 2: Extract document-level metadata
            doc_metadata = metadata_extractor.extract_document_metadata(document)

            # Step 3: Enhance each chunk with metadata
            enhanced_chunks = []
            for chunk in chunks:
                enhanced_chunk = chunk.copy()
                enhanced_chunk['metadata'] = metadata_extractor.enhance_chunk_metadata(chunk, doc_metadata)
                enhanced_chunks.append(enhanced_chunk)

            # Step 4: Embed the chunks
            embedded_chunks = self.embed_chunks(enhanced_chunks)

            log_info(f"Successfully processed document for embedding", extra={
                "document_id": document.get('document_id'),
                "original_chunks": len(chunks),
                "embedded_chunks": len(embedded_chunks)
            })

            return embedded_chunks

        except Exception as e:
            log_error(f"Error processing document for embedding: {str(e)}", extra={
                "document_id": document.get('document_id')
            })
            raise

    def validate_embeddings(self, embedded_chunks: List[Dict]) -> bool:
        """
        Validate that all embeddings have the correct dimensions
        """
        if not embedded_chunks:
            return True

        expected_dimension = config.VECTOR_SIZE

        for chunk in embedded_chunks:
            embedding = chunk.get('embedding')
            if not embedding:
                log_error("Missing embedding in chunk", extra={"chunk_id": chunk.get('chunk_id')})
                return False

            if len(embedding) != expected_dimension:
                log_error(f"Embedding dimension mismatch", extra={
                    "expected": expected_dimension,
                    "actual": len(embedding),
                    "chunk_id": chunk.get('chunk_id')
                })
                return False

        return True

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        """
        # Convert to numpy arrays for calculation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

# Create a singleton instance
embedding_pipeline = EmbeddingPipeline()