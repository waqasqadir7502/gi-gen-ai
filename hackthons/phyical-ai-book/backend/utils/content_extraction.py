import os
from pathlib import Path
from typing import List, Dict
from .markdown_parser import markdown_parser

class ContentExtractor:
    def __init__(self, docs_path: str = "../../docs"):
        self.docs_path = Path(docs_path).resolve()

    def extract_from_docs_folder(self) -> List[Dict]:
        """
        Extract content from the /docs folder as specified in the requirements
        """
        if not self.docs_path.exists():
            raise FileNotFoundError(f"Docs folder not found at {self.docs_path}")

        # Parse all markdown files in the docs directory
        parsed_documents = markdown_parser.parse_directory(str(self.docs_path))

        # Add document IDs for tracking
        for i, doc in enumerate(parsed_documents):
            doc['document_id'] = f"doc_{i}_{hash(doc['metadata']['file_path'])}"

        return parsed_documents

    def extract_from_sitemap_fallback(self, sitemap_url: str = None) -> List[Dict]:
        """
        Fallback method to extract content from HTML pages via sitemap.xml
        This is a placeholder implementation - in a real system you'd fetch and parse the sitemap
        """
        # This is out of scope for v1 as per requirements
        print("Sitemap extraction is out of scope for v1")
        return []

    def get_document_by_path(self, file_path: str) -> Dict:
        """
        Extract content from a specific file path
        """
        return markdown_parser.parse_file(file_path)

# Create a singleton instance
content_extractor = ContentExtractor()