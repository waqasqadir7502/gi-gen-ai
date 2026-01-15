import sys
from pathlib import Path
# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy import Column, Integer, String, DateTime, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Document(Base):
    """
    Document model for storing document metadata in Neon database
    """
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String, unique=True, index=True, nullable=False)
    file_path = Column(String, nullable=False)
    relative_path = Column(String, nullable=False)
    title = Column(String, nullable=True)
    content_hash = Column(String, nullable=False)
    content_length = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    source_url = Column(String, nullable=True)
    metadata_json = Column(Text, nullable=True)  # For storing additional metadata as JSON

    # Index for fast lookups
    __table_args__ = (
        Index('idx_doc_document_id', 'document_id'),
        Index('idx_doc_content_hash', 'content_hash'),
        Index('idx_doc_file_path', 'file_path'),
    )

class Chunk(Base):
    """
    Chunk model for storing chunk metadata in Neon database
    This links to document chunks but stores metadata separately from vector embeddings in Qdrant
    """
    __tablename__ = 'chunks'

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(String, unique=True, index=True, nullable=False)  # This would match Qdrant point ID
    document_id = Column(String, nullable=False)  # References the document
    chunk_index = Column(Integer, nullable=True)
    content_preview = Column(Text, nullable=True)  # First 500 chars for quick reference
    content_length = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(Text, nullable=True)  # Additional metadata as JSON

    # Index for fast lookups
    __table_args__ = (
        Index('idx_chunk_id', 'chunk_id'),
        Index('idx_chunk_document_id', 'document_id'),
    )