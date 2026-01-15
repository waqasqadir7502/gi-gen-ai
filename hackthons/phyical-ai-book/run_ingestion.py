#!/usr/bin/env python3
"""
Temporary script to run the ingestion process
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import the required modules
from backend.utils.content_extraction import content_extractor
from backend.pipelines.indexing_pipeline import indexing_pipeline
from backend.utils.logger import log_info, log_error
from backend.config import config

def main():
    log_info("Starting documentation ingestion process", extra={
        "docs_path": "docs",
        "batch_size": 10,
        "skip_duplicates": True,
        "dry_run": False
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
        extractor.docs_path = Path("docs").resolve()

        if not extractor.docs_path.exists():
            log_error(f"Docs folder not found at {extractor.docs_path}")
            return False

        parsed_documents = extractor.extract_from_docs_folder()
        log_info(f"Extracted {len(parsed_documents)} documents from {extractor.docs_path}")

        if not parsed_documents:
            print("No documents found to index")
            return True

        # Index documents
        log_info("Starting indexing process...")
        results = indexing_pipeline.index_documents_batch(parsed_documents, 10)

        # Log results
        log_info("Indexing completed", extra=results)

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
        print("Ingestion interrupted by user")
        return False

    except Exception as e:
        log_error(f"Error during ingestion: {str(e)}", extra={
            "error_type": type(e).__name__
        })
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)