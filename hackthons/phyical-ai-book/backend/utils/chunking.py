import re
from typing import List, Dict, Tuple
from .logger import log_info, log_warning
import tiktoken

class ChunkingAlgorithm:
    def __init__(self, target_size: int = 768, overlap: int = 150):
        """
        Initialize chunking algorithm with target size and overlap

        Args:
            target_size: Target chunk size in tokens (default 768)
            overlap: Overlap size in tokens (default 150)
        """
        self.target_size = target_size
        self.overlap = overlap
        # Using cl100k_base encoding which is used by gpt-3.5-turbo, gpt-4, etc.
        # For Cohere embeddings, we'll approximate tokenization
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for the given text
        """
        try:
            return len(self.encoder.encode(text))
        except Exception:
            # Fallback: rough estimation based on word count
            return len(text.split()) // 0.75  # Approximate average token/word ratio

    def chunk_by_semantic_boundaries(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks respecting semantic boundaries
        """
        if not text.strip():
            return []

        # Split text into sentences or paragraphs
        sentences = self._split_by_sentence(text)

        chunks = []
        current_chunk = ""
        current_metadata = metadata.copy() if metadata else {}
        chunk_start_pos = 0

        for i, sentence in enumerate(sentences):
            sentence_token_count = self.estimate_token_count(sentence)
            current_token_count = self.estimate_token_count(current_chunk)

            # If adding this sentence would exceed target size
            if current_token_count + sentence_token_count > self.target_size and current_chunk:
                # Save the current chunk
                chunks.append({
                    "content": current_chunk.strip(),
                    "metadata": current_metadata,
                    "token_count": current_token_count,
                    "start_pos": chunk_start_pos,
                    "end_pos": chunk_start_pos + len(current_chunk)
                })

                # Start a new chunk with overlap from the previous chunk
                if self.overlap > 0:
                    # Get the end portion of the current chunk for overlap
                    overlap_text = self._get_overlap_text(current_chunk, self.overlap)
                    current_chunk = overlap_text + " " + sentence
                    chunk_start_pos = len(current_chunk) - len(overlap_text + " " + sentence)
                else:
                    current_chunk = sentence
                    chunk_start_pos = text.find(sentence, chunk_start_pos)
            else:
                # Add sentence to current chunk
                current_chunk += " " + sentence if current_chunk else sentence

        # Add the final chunk if it has content
        if current_chunk.strip():
            current_token_count = self.estimate_token_count(current_chunk)
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": current_metadata,
                "token_count": current_token_count,
                "start_pos": chunk_start_pos,
                "end_pos": chunk_start_pos + len(current_chunk)
            })

        log_info(f"Created {len(chunks)} chunks from text", extra={
            "original_length": len(text),
            "target_size": self.target_size,
            "overlap": self.overlap
        })

        return chunks

    def _split_by_sentence(self, text: str) -> List[str]:
        """
        Split text into sentences while preserving context
        """
        # This is a simple sentence splitter
        # In production, consider using nltk or spacy for better sentence segmentation
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z].)\.(?=\s+[A-Z])|(?<=[.!?])\s+', text)

        # Clean up the sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _get_overlap_text(self, text: str, token_count: int) -> str:
        """
        Get the end portion of text that corresponds to approximately token_count tokens
        """
        # Start from the end and work backwards
        words = text.split()
        estimated_tokens_per_word = 1.3  # Approximate ratio

        # Estimate how many words we need
        approx_words_needed = int(token_count / estimated_tokens_per_word)

        # Get the last N words
        overlap_words = words[-approx_words_needed:] if len(words) >= approx_words_needed else words

        overlap_text = " ".join(overlap_words)

        # Adjust if the token count is too far off
        actual_tokens = self.estimate_token_count(overlap_text)
        if actual_tokens > token_count:
            # Reduce the text until we're close to the target
            while actual_tokens > token_count and len(overlap_words) > 1:
                overlap_words = overlap_words[1:]
                overlap_text = " ".join(overlap_words)
                actual_tokens = self.estimate_token_count(overlap_text)
        elif actual_tokens < token_count * 0.5:
            # Expand if we're too small
            additional_needed = int((token_count - actual_tokens) / estimated_tokens_per_word)
            start_idx = max(0, len(words) - len(overlap_words) - additional_needed)
            overlap_words = words[start_idx:]
            overlap_text = " ".join(overlap_words)

        return overlap_text

    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Chunk an entire document with metadata preservation
        """
        content = document.get('content', '')
        metadata = document.get('metadata', {})

        # Add document-specific metadata to each chunk
        document_metadata = {
            'document_id': document.get('document_id', ''),
            'file_path': metadata.get('file_path', ''),
            'relative_path': metadata.get('relative_path', ''),
            'source_title': metadata.get('title', ''),
            'source_heading': metadata.get('headings', [{}])[0].get('text', '') if metadata.get('headings') else '',
        }

        chunks = self.chunk_by_semantic_boundaries(content, document_metadata)

        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk['chunk_id'] = f"{document.get('document_id', 'unknown')}_{i}"
            chunk['chunk_index'] = i
            chunk['total_chunks'] = len(chunks)

        log_info(f"Document chunked into {len(chunks)} pieces", extra={
            "document_id": document.get('document_id'),
            "original_tokens": self.estimate_token_count(content),
            "chunks_created": len(chunks)
        })

        return chunks

# Create a singleton instance
chunker = ChunkingAlgorithm()