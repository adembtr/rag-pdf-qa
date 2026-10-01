"""
Configuration settings for the RAG (Retrieval-Augmented Generation) system.

This module defines all configurable parameters for the RAG pipeline including
text chunking settings, embedding model selection, file paths, and query parameters.
"""

from pathlib import Path

# =============================================================================
# Text Chunking Settings
# =============================================================================

CHUNK_SIZE: int = 500
"""
Number of characters per text chunk.

Smaller chunks provide more precise retrieval but may lose context.
Larger chunks preserve context but may include irrelevant information.
500 characters is a balanced default for most use cases.
"""

CHUNK_OVERLAP: int = 50
"""
Number of overlapping characters between consecutive chunks.

Overlap ensures that information at chunk boundaries is not lost and
helps maintain semantic continuity across chunks.
"""

# =============================================================================
# Embedding Model Settings
# =============================================================================

EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
"""
Pre-trained sentence transformer model for generating text embeddings.

This multilingual model supports 50+ languages including Turkish, English,
German, French, Spanish, and many others. It provides a good balance between
performance and model size (420MB, 384-dimensional embeddings).
"""

# =============================================================================
# Directory Paths
# =============================================================================

# Base directory is the parent of this config file's location
_BASE_DIR: Path = Path(__file__).parent

PDF_DIR: Path = _BASE_DIR / ".." / "data" / "pdfs"
"""
Directory where PDF files are stored for processing.

The RAG system will read PDF documents from this location,
extract text, and create embeddings for retrieval.
"""

DB_DIR: Path = _BASE_DIR / ".." / "data" / "db"
"""
Directory where ChromaDB stores its persistent data.

This includes the vector index and metadata for all embedded documents.
The database is created automatically if it doesn't exist.
"""

# =============================================================================
# ChromaDB Settings
# =============================================================================

COLLECTION_NAME: str = "kitaplar"
"""
Name of the ChromaDB collection for storing document embeddings.

A collection is a named grouping of embeddings and their associated metadata.
'kitaplar' means 'books' in Turkish.
"""

# =============================================================================
# Query Settings
# =============================================================================

TOP_K: int = 5
"""
Default number of results to return from similarity queries.

This determines how many of the most relevant text chunks are retrieved
when querying the vector database. Adjust based on your use case:
- Lower values (3-5): More focused, faster responses
- Higher values (10-20): More comprehensive context, potentially noisier
"""