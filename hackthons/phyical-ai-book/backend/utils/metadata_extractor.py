from typing import Dict, List, Any
import hashlib
from datetime import datetime
from urllib.parse import urlparse
import os

class MetadataExtractor:
    def __init__(self):
        pass

    def extract_document_metadata(self, document: Dict) -> Dict[str, Any]:
        """
        Extract and enhance metadata from a document
        """
        raw_metadata = document.get('metadata', {})
        content = document.get('content', '')

        # Extract document-level metadata
        doc_metadata = {
            'title': raw_metadata.get('title', ''),
            'description': raw_metadata.get('description', ''),
            'author': raw_metadata.get('author', ''),
            'tags': raw_metadata.get('tags', []),
            'category': raw_metadata.get('category', ''),
            'source_type': 'markdown',
            'content_length': len(content),
            'word_count': len(content.split()),
            'created_at': raw_metadata.get('created_time', datetime.now().isoformat()),
            'updated_at': raw_metadata.get('modified_time', datetime.now().isoformat()),
            'file_path': raw_metadata.get('file_path', ''),
            'relative_path': raw_metadata.get('relative_path', ''),
            'file_name': raw_metadata.get('file_name', ''),
        }

        # Generate content-based identifier
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        doc_metadata['content_hash'] = content_hash

        # Extract section-level context from headings
        headings = raw_metadata.get('headings', [])
        doc_metadata['headings'] = headings
        doc_metadata['heading_count'] = len(headings)

        # Extract first heading as context
        if headings:
            first_heading = headings[0]
            doc_metadata['primary_heading'] = first_heading['text']
            doc_metadata['primary_heading_level'] = first_heading['level']

        # Determine content type classification
        doc_metadata['content_type'] = self._classify_content_type(content, headings)

        # Create source identifier with precise location references
        doc_metadata['source_id'] = self._create_source_identifier(doc_metadata)

        # Create mapping to original content locations with anchor links
        doc_metadata['anchor_links'] = self._create_anchor_links(headings, doc_metadata)

        return doc_metadata

    def _classify_content_type(self, content: str, headings: List[Dict]) -> str:
        """
        Classify content type based on content and headings
        """
        content_lower = content.lower()

        # Check for code blocks
        if '```' in content or '    ' in content[:100]:  # Check beginning for indented code
            return 'code'

        # Check for examples or tutorials
        if any(keyword in content_lower for keyword in ['example', 'tutorial', 'how to', 'step', 'guide']):
            return 'tutorial'

        # Check for API documentation
        if any(keyword in content_lower for keyword in ['api', 'endpoint', 'request', 'response', 'parameter']):
            return 'api_doc'

        # Check for configuration files
        if any(keyword in content_lower for keyword in ['config', 'configuration', 'setting', 'environment']):
            return 'configuration'

        # Check for general documentation
        if any(keyword in content_lower for keyword in ['documentation', 'overview', 'introduction', 'summary']):
            return 'documentation'

        # Default to text
        return 'text'

    def _create_source_identifier(self, metadata: Dict) -> str:
        """
        Create a unique source identifier with precise location references
        """
        file_path = metadata.get('file_path', '')
        title = metadata.get('title', '')
        primary_heading = metadata.get('primary_heading', '')

        # Create identifier combining file path, title and primary heading
        identifier_parts = [file_path]
        if title:
            identifier_parts.append(title)
        if primary_heading:
            identifier_parts.append(primary_heading)

        # Create a hash of the identifier parts to ensure uniqueness
        identifier_str = '::'.join(identifier_parts)
        identifier_hash = hashlib.sha256(identifier_str.encode('utf-8')).hexdigest()[:16]

        return f"src_{identifier_hash}"

    def _create_anchor_links(self, headings: List[Dict], metadata: Dict) -> List[Dict]:
        """
        Create anchor links for headings with location references
        """
        anchor_links = []

        for heading in headings:
            anchor_links.append({
                'text': heading['text'],
                'level': heading['level'],
                'anchor': self._create_anchor_name(heading['text']),
                'source_path': metadata.get('relative_path', ''),
                'source_file': metadata.get('file_name', '')
            })

        return anchor_links

    def _create_anchor_name(self, heading_text: str) -> str:
        """
        Create a URL-friendly anchor name from heading text
        """
        # Convert to lowercase and replace spaces with hyphens
        anchor = heading_text.lower().replace(' ', '-').replace('_', '-')
        # Remove special characters
        anchor = ''.join(c for c in anchor if c.isalnum() or c == '-')
        # Remove multiple consecutive hyphens
        while '--' in anchor:
            anchor = anchor.replace('--', '-')

        return anchor

    def enhance_chunk_metadata(self, chunk: Dict, document_metadata: Dict) -> Dict:
        """
        Enhance chunk-level metadata with document context
        """
        enhanced_metadata = chunk.get('metadata', {}).copy()

        # Inherit document-level metadata
        enhanced_metadata.update({
            'document_title': document_metadata.get('title', ''),
            'document_path': document_metadata.get('relative_path', ''),
            'document_source_id': document_metadata.get('source_id', ''),
            'content_type': document_metadata.get('content_type', 'text'),
            'primary_heading': document_metadata.get('primary_heading', ''),
            'chunk_id': chunk.get('chunk_id', ''),
            'chunk_index': chunk.get('chunk_index', 0),
            'total_chunks': chunk.get('total_chunks', 1),
        })

        # Add position information
        enhanced_metadata['start_pos'] = chunk.get('start_pos', 0)
        enhanced_metadata['end_pos'] = chunk.get('end_pos', 0)

        # Add content statistics
        enhanced_metadata['token_count'] = chunk.get('token_count', 0)
        enhanced_metadata['content_length'] = len(chunk.get('content', ''))

        return enhanced_metadata

    def extract_from_text(self, text: str, additional_metadata: Dict = None) -> Dict[str, Any]:
        """
        Extract metadata directly from text content
        """
        metadata = {
            'content_length': len(text),
            'word_count': len(text.split()),
            'char_count': len(text),
            'line_count': len(text.splitlines()),
            'content_hash': hashlib.md5(text.encode('utf-8')).hexdigest(),
            'extracted_at': datetime.now().isoformat(),
        }

        if additional_metadata:
            metadata.update(additional_metadata)

        return metadata

# Create a singleton instance
metadata_extractor = MetadataExtractor()