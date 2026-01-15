#!/usr/bin/env python3
"""
Ingestion script for the Physical AI Book RAG Chatbot
This script processes markdown files from the /docs folder and indexes them in Qdrant
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.utils.content_extraction import content_extractor
from backend.pipelines.indexing_pipeline import indexing_pipeline
from backend.utils.logger import log_info, log_error, log_warning
from backend.config import config
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Ingest Physical AI Book documentation into vector store")
    parser.add_argument("--docs-path", type=str, default="../../docs", help="Path to docs folder")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for indexing")
    parser.add_argument("--skip-duplicates", action="store_true", help="Skip duplicate content")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually index, just show what would be done")

    args = parser.parse_args()

    log_info("Starting documentation ingestion process", extra={
        "docs_path": args.docs_path,
        "batch_size": args.batch_size,
        "skip_duplicates": args.skip_duplicates,
        "dry_run": args.dry_run
    })

    try:
        # Validate configuration
        if not config.COHERE_API_KEY:
            log_error("COHERE_API_KEY not found in environment")
            return False

        if not config.QDRANT_URL or not config.QDRANT_API_KEY:
            log_error("QDRANT credentials not found in environment")
            return False

        # Extract content from docs folder
        log_info("Extracting content from docs folder...")

        # Update the extractor with the provided path
        extractor = content_extractor
        extractor.docs_path = Path(args.docs_path).resolve()

        if not extractor.docs_path.exists():
            log_error(f"Docs folder not found at {extractor.docs_path}")
            return False

        parsed_documents = extractor.extract_from_docs_folder()
        log_info(f"Extracted {len(parsed_documents)} documents from {extractor.docs_path}")

        if not parsed_documents:
            log_warning("No documents found to index")
            return True

        # Process documents for indexing
        if args.dry_run:
            log_info("DRY RUN: Would process the following documents:", extra={
                "documents": [doc['metadata']['file_name'] for doc in parsed_documents]
            })
            return True

        # Index documents
        log_info("Starting indexing process...")

        # Use the appropriate indexing method based on duplicate check preference
        if args.skip_duplicates:
            log_info("Duplicate checking enabled")
            results = indexing_pipeline.index_documents_batch(parsed_documents, args.batch_size)
        else:
            results = indexing_pipeline.index_documents_batch(parsed_documents, args.batch_size)

        # Log results
        log_info("Indexing completed", extra=results)

        # Verification
        log_info("Verifying collection contents...")
        collection_info = indexing_pipeline.collection_name
        log_info(f"Documents indexed in collection: {collection_info}")

        # Show summary
        print("\n" + "="*50)
        print("INGESTION SUMMARY")
        print("="*50)
        print(f"Total documents processed: {results['total_documents']}")
        print(f"Successfully indexed: {results['successful']}")
        print(f"Failed to index: {results['failed']}")
        print(f"Success rate: {results['success_rate']*100:.2f}%")

        if results['failed'] > 0:
            print(f"\nFailed documents:")
            for failed_doc in results['failed_documents']:
                print(f"  - {failed_doc.get('document_id', 'Unknown')}")

        print("="*50)

        return results['success_rate'] == 1.0  # Return True if all succeeded

    except KeyboardInterrupt:
        log_warning("Ingestion interrupted by user")
        return False

    except Exception as e:
        log_error(f"Error during ingestion: {str(e)}", extra={
            "error_type": type(e).__name__
        })
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)