from typing import List, Dict, Any
from ..clients.qdrant_client import qdrant_client
from ..config import config
from ..utils.logger import log_info, log_error, log_warning
from ..utils.metadata_extractor import metadata_extractor
import hashlib
import uuid
from datetime import datetime

class IndexingPipeline:
    def __init__(self):
        self.collection_name = config.COLLECTION_NAME
        self.vector_size = config.VECTOR_SIZE

    def upsert_chunks_to_qdrant(self, embedded_chunks: List[Dict]) -> bool:
        """
        Upsert embedded chunks to Qdrant collection and save metadata to Neon database
        """
        if not embedded_chunks:
            log_warning("No embedded chunks to upsert")
            return True

        try:
            # Prepare vectors, payloads, and IDs for upsert
            vectors = []
            payloads = []
            ids = []

            # Process chunks for both Qdrant and database
            for chunk in embedded_chunks:
                embedding = chunk.get('embedding')
                if not embedding:
                    log_error("Missing embedding in chunk", extra={"chunk_id": chunk.get('chunk_id')})
                    continue

                # Generate a consistent ID for both Qdrant and database correlation
                chunk_id = str(uuid.uuid4())

                # Prepare payload with metadata
                payload = {
                    'content': chunk['content'],
                    'document_id': chunk['metadata'].get('document_id', ''),
                    'file_path': chunk['metadata'].get('file_path', ''),
                    'relative_path': chunk['metadata'].get('relative_path', ''),
                    'source_title': chunk['metadata'].get('document_title', ''),
                    'source_heading': chunk['metadata'].get('primary_heading', ''),
                    'content_type': chunk['metadata'].get('content_type', 'text'),
                    'chunk_index': chunk['metadata'].get('chunk_index', 0),
                    'total_chunks': chunk['metadata'].get('total_chunks', 1),
                    'token_count': chunk['metadata'].get('token_count', 0),
                    'content_length': chunk['metadata'].get('content_length', 0),
                    'content_hash': chunk['metadata'].get('content_hash', ''),
                    'indexed_at': datetime.now().isoformat(),
                    'chunk_id': chunk_id  # Add the ID for database correlation
                }

                # Add any additional metadata
                for key, value in chunk['metadata'].items():
                    if key not in payload:
                        payload[key] = value

                vectors.append(embedding)
                payloads.append(payload)
                ids.append(chunk_id)  # Use the same ID for Qdrant

            if not vectors:
                log_warning("No valid vectors to upsert after processing")
                return True

            # Upsert to Qdrant
            success = qdrant_client.upsert_vectors(vectors, payloads, ids)

            if success:
                # Save metadata to Neon database as well (only if db_manager is available)
                try:
                    from ..utils.db_connection import db_manager
                    if db_manager:
                        self._save_metadata_to_database(embedded_chunks, ids)
                except ImportError:
                    # Database not configured, just log and continue
                    log_info("Database not configured, skipping metadata storage", extra={
                        "collection_name": self.collection_name,
                        "vectors_upserted": len(vectors)
                    })

                log_info(f"Successfully upserted {len(vectors)} vectors to Qdrant", extra={
                    "collection_name": self.collection_name,
                    "vectors_upserted": len(vectors)
                })
                return True
            else:
                log_error("Failed to upsert vectors to Qdrant")
                return False

        except Exception as e:
            log_error(f"Error during Qdrant upsert: {str(e)}", extra={
                "collection_name": self.collection_name
            })
            return False

    def _save_metadata_to_database(self, embedded_chunks: List[Dict], qdrant_ids: List[str]):
        """
        Save document and chunk metadata to Neon database
        """
        try:
            from ..utils.db_connection import db_manager
            if not hasattr(db_manager, 'engine') or not db_manager.engine:
                log_warning("Neon database not configured, skipping metadata storage")
                return
        except ImportError:
            log_warning("Database module not available, skipping metadata storage")
            return

        try:
            session = db_manager.get_session()

            # Import Document and Chunk models here to avoid circular imports
            from ..models.document_model import Document, Chunk
            from sqlalchemy.exc import SQLAlchemyError

            for i, chunk in enumerate(embedded_chunks):
                if i >= len(qdrant_ids):
                    break

                chunk_id = qdrant_ids[i]
                metadata = chunk.get('metadata', {})

                # Create or update document record
                document = session.query(Document).filter(Document.document_id == metadata.get('document_id')).first()
                if not document:
                    document = Document(
                        document_id=metadata.get('document_id', ''),
                        file_path=metadata.get('file_path', ''),
                        relative_path=metadata.get('relative_path', ''),
                        title=metadata.get('document_title', ''),
                        content_hash=metadata.get('content_hash', ''),
                        content_length=metadata.get('content_length', 0),
                        token_count=metadata.get('token_count', 0),
                        source_url=metadata.get('source_url', ''),
                        metadata_json=str(metadata)  # Store additional metadata as JSON string
                    )
                    session.add(document)

                # Create chunk record
                chunk_record = Chunk(
                    chunk_id=chunk_id,  # This matches the Qdrant ID for correlation
                    document_id=metadata.get('document_id', ''),
                    chunk_index=metadata.get('chunk_index', 0),
                    content_preview=chunk['content'][:500] if chunk.get('content') else '',  # First 500 chars
                    content_length=len(chunk.get('content', '')),
                    token_count=metadata.get('token_count', 0),
                    metadata_json=str(metadata)  # Store chunk metadata as JSON string
                )
                session.add(chunk_record)

            session.commit()
            log_info(f"Saved metadata for {len(embedded_chunks)} chunks to Neon database")

        except Exception as e:
            log_error(f"Error saving metadata to Neon database: {str(e)}")
            if 'session' in locals():
                session.rollback()
        finally:
            if 'session' in locals():
                session.close()

    def index_document(self, document: Dict) -> bool:
        """
        Process and index a single document end-to-end: chunk, embed, and upsert to Qdrant
        """
        try:
            # Step 1: Process document through embedding pipeline
            # We'll import here to avoid circular imports
            from .embedding_pipeline import embedding_pipeline
            embedded_chunks = embedding_pipeline.process_document_for_embedding(document)

            # Step 2: Validate embeddings
            if not embedding_pipeline.validate_embeddings(embedded_chunks):
                log_error("Embedding validation failed", extra={"document_id": document.get('document_id')})
                return False

            # Step 3: Upsert to Qdrant
            success = self.upsert_chunks_to_qdrant(embedded_chunks)

            if success:
                log_info("Document successfully indexed", extra={
                    "document_id": document.get('document_id'),
                    "chunks_indexed": len(embedded_chunks)
                })

            return success

        except Exception as e:
            log_error(f"Error indexing document: {str(e)}", extra={
                "document_id": document.get('document_id')
            })
            return False

    def index_documents_batch(self, documents: List[Dict], batch_size: int = 10) -> Dict[str, Any]:
        """
        Index multiple documents in batches for efficiency
        """
        total_docs = len(documents)
        successful = 0
        failed = 0
        failed_docs = []

        log_info(f"Starting batch indexing of {total_docs} documents", extra={
            "batch_size": batch_size
        })

        for i, document in enumerate(documents):
            try:
                success = self.index_document(document)
                if success:
                    successful += 1
                else:
                    failed += 1
                    failed_docs.append({
                        "document_id": document.get('document_id'),
                        "index": i
                    })

                # Log progress every 10 documents
                if (i + 1) % 10 == 0:
                    log_info(f"Batch indexing progress", extra={
                        "processed": i + 1,
                        "total": total_docs,
                        "successful": successful,
                        "failed": failed
                    })

            except Exception as e:
                failed += 1
                failed_docs.append({
                    "document_id": document.get('document_id'),
                    "index": i,
                    "error": str(e)
                })
                log_error(f"Error indexing document at index {i}: {str(e)}", extra={
                    "document_id": document.get('document_id')
                })

        result = {
            "total_documents": total_docs,
            "successful": successful,
            "failed": failed,
            "failed_documents": failed_docs,
            "success_rate": successful / total_docs if total_docs > 0 else 0
        }

        log_info("Batch indexing completed", extra=result)

        return result

    def check_duplicate_content(self, content: str) -> bool:
        """
        Check if content already exists in the collection (basic duplicate detection)
        """
        try:
            # Create a hash of the content to check for duplicates
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

            # Search for documents with the same content hash
            # This is a simple approach - in production, you might want to use
            # semantic similarity instead of exact hash matching
            search_results = qdrant_client.search(
                query_vector=[0.0] * self.vector_size,  # Dummy vector for filtering only
                top_k=1,
                filters={"content_hash": content_hash}
            )

            return len(search_results) > 0

        except Exception as e:
            log_error(f"Error checking for duplicates: {str(e)}")
            # If there's an error, assume it's not a duplicate to be safe
            return False

    def index_document_with_duplicate_check(self, document: Dict) -> bool:
        """
        Index a document after checking for duplicates
        """
        try:
            # For duplicate checking, we'll check the original content
            content = document.get('content', '')

            # Check if this content already exists
            if self.check_duplicate_content(content):
                log_info("Document is a duplicate, skipping indexing", extra={
                    "document_id": document.get('document_id')
                })
                return True  # Consider it successful since it's already indexed

            # If not a duplicate, proceed with indexing
            return self.index_document(document)

        except Exception as e:
            log_error(f"Error during duplicate-checked indexing: {str(e)}", extra={
                "document_id": document.get('document_id')
            })
            return False

# Create a singleton instance
indexing_pipeline = IndexingPipeline()