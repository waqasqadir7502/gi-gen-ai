import markdown
import mistune
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path

class MarkdownParser:
    def __init__(self):
        # Initialize mistune markdown parser
        self.markdown_renderer = mistune.create_markdown(
            renderer='html',
            plugins=['strikethrough', 'footnotes', 'table', 'url']
        )

    def parse_file(self, file_path: str) -> Dict:
        """
        Parse a markdown file and extract content, metadata, and structure
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract metadata from frontmatter if present
        metadata = self._extract_frontmatter(content)

        # Remove frontmatter from content
        content_without_frontmatter = self._remove_frontmatter(content)

        # Parse the markdown content
        html_content = self.markdown_renderer(content_without_frontmatter)

        # Extract text content while preserving structure
        soup = BeautifulSoup(html_content, 'html.parser')
        text_content = soup.get_text(separator=' ', strip=True)

        # Extract headings for structure
        headings = self._extract_headings(soup)

        # Get file-specific metadata
        file_stats = os.stat(file_path)
        file_metadata = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'relative_path': str(Path(file_path).relative_to(Path(file_path).parent.parent.parent)),  # Adjust based on project structure
            'size_bytes': file_stats.st_size,
            'modified_time': file_stats.st_mtime,
            'created_time': file_stats.st_ctime,
        }

        # Combine all metadata
        combined_metadata = {**metadata, **file_metadata, 'headings': headings}

        return {
            'content': text_content,
            'raw_content': content_without_frontmatter,
            'html_content': html_content,
            'metadata': combined_metadata,
            'headings': headings
        }

    def _extract_frontmatter(self, content: str) -> Dict:
        """
        Extract YAML frontmatter from markdown content
        """
        frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if match:
            frontmatter_str = match.group(1)
            # Simple parsing of frontmatter - in a real implementation, use pyyaml
            metadata = {}
            for line in frontmatter_str.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    metadata[key] = value
            return metadata
        return {}

    def _remove_frontmatter(self, content: str) -> str:
        """
        Remove YAML frontmatter from content
        """
        frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
        return re.sub(frontmatter_pattern, '', content, 1, re.DOTALL)

    def _extract_headings(self, soup) -> List[Dict]:
        """
        Extract headings with their levels and positions
        """
        headings = []
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            level = int(heading.name[1])  # h1 -> 1, h2 -> 2, etc.
            text = heading.get_text(strip=True)

            headings.append({
                'level': level,
                'text': text,
                'element': heading
            })
        return headings

    def parse_directory(self, directory_path: str, extensions: List[str] = ['.md', '.markdown']) -> List[Dict]:
        """
        Parse all markdown files in a directory recursively
        """
        all_parsed_files = []

        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext.lower()) for ext in extensions):
                    file_path = os.path.join(root, file)
                    try:
                        parsed_file = self.parse_file(file_path)
                        all_parsed_files.append(parsed_file)
                    except Exception as e:
                        print(f"Error parsing {file_path}: {str(e)}")

        return all_parsed_files

    def extract_sections(self, content: str, max_section_length: int = 2000) -> List[str]:
        """
        Extract sections from content based on headings and length constraints
        """
        # This is a simplified version - a more sophisticated approach would
        # respect semantic boundaries and document structure
        sections = []

        # Split content by common section separators
        paragraphs = re.split(r'\n\s*\n+', content)

        current_section = ""
        for paragraph in paragraphs:
            if len(current_section) + len(paragraph) < max_section_length:
                current_section += paragraph + "\n\n"
            else:
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = paragraph + "\n\n"

        # Add the last section if it exists
        if current_section.strip():
            sections.append(current_section.strip())

        return sections

# Create a singleton instance
markdown_parser = MarkdownParser()