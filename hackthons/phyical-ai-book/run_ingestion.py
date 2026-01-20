#!/usr/bin/env python3
"""
Ingestion script for Physical AI Book documentation - safe for Vercel & API usage
"""

import os
from pathlib import Path
import sys

from backend.utils.content_extraction import content_extractor
from backend.pipelines.indexing_pipeline import indexing_pipeline
from backend.utils.logger import log_info, log_error
from backend.config import config
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse


def run_ingestion(docs_path: str = "docs", batch_size: int = 10, skip_duplicates: bool = True, dry_run: bool = False) -> bool:
    """
    Main ingestion function - can be called from script or API
    Returns True on success, False on failure (no sys.exit!)
    """
    log_info("Starting documentation ingestion process", extra={
        "docs_path": docs_path,
        "batch_size": batch_size,
        "skip_duplicates": skip_duplicates,
        "dry_run": dry_run
    })

    try:
        # Validate config
        if not config.COHERE_API_KEY:
            log_error("COHERE_API_KEY not found in environment")
            return False

        if not config.QDRANT_URL or not config.QDRANT_API_KEY:
            log_error("QDRANT credentials not found in environment")
            return False

        # Resolve docs path (use env var or fallback)
        docs_root = Path(os.getenv("DOCS_PATH", docs_path)).resolve()
        if not docs_root.exists():
            log_error(f"Docs folder not found at {docs_root}")
            return False

        log_info(f"Extracting content from {docs_root}...")

        # Configure extractor
        extractor.docs_path = docs_root
        parsed_documents = extractor.extract_from_docs_folder()

        log_info(f"Extracted {len(parsed_documents)} documents")

        if not parsed_documents:
            log_info("No documents found to index")
            return True  # Success - nothing to do

        if dry_run:
            log_info("Dry run mode - skipping actual indexing")
            return True

        # Indexing
        log_info("Starting indexing process...")
        results = indexing_pipeline.index_documents_batch(
            parsed_documents,
            batch_size=batch_size,
            skip_duplicates=skip_duplicates
        )

        # Summary
        log_info("Indexing completed", extra=results)

        print("\n" + "="*50)
        print("INGESTION SUMMARY")
        print("="*50)
        print(f"Total documents processed: {results['total_documents']}")
        print(f"Successfully indexed: {results['successful']}")
        print(f"Failed to index: {results['failed']}")
        print(f"Success rate: {results['success_rate']*100:.2f}%")

        if results['failed'] > 0:
            print("\nFailed documents:")
            for failed_doc in results['failed_documents']:
                print(f"  - {failed_doc.get('document_id', 'Unknown')}")

        print("="*50)

        return results['success_rate'] == 1.0

    except KeyboardInterrupt:
        log_info("Ingestion interrupted by user")
        return False

    except Exception as e:
        log_error(f"Critical error during ingestion: {str(e)}", extra={
            "error_type": type(e).__name__,
            "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__))
        })
        return False


def ensure_collection_exists():
    """Safe, idempotent collection check/create - used in API startup or ingestion"""
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
    )

    collection_name = "physical-ai-book-v1"

    try:
        if not client.has_collection(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config={"size": 1024, "distance": "Cosine"}
            )
            log_info(f"Created Qdrant collection: {collection_name}")
        else:
            log_info(f"Collection {collection_name} already exists - skipping creation")
    except UnexpectedResponse as e:
        if e.status_code == 409:  # Conflict = already exists
            log_info(f"Collection already exists (409) - safe to continue")
        else:
            log_error(f"Qdrant collection error: {e}")
            raise
    except Exception as e:
        log_error(f"Failed to ensure collection: {str(e)}")
        raise


if __name__ == "__main__":
    # When run as script: safe mode, no exit
    success = run_ingestion()
    print(f"Ingestion finished with success: {success}")
    # NO sys.exit() here - just print status