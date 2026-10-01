"""
PDF Ingestion Script for RAG System.

This script processes PDF files from the configured directory, extracts text,
splits it into chunks, generates embeddings, and stores them in ChromaDB
for later retrieval.

Usage:
    python ingest.py
"""

import re
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DB_DIR,
    EMBEDDING_MODEL,
    PDF_DIR,
)


def extract_text_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    """
    Extract text from each page of a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of tuples containing (page_number, page_text).
        Page numbers are 1-indexed. Pages with no text are skipped.

    Raises:
        RuntimeError: If the PDF cannot be opened or processed.
    """
    pages = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF '{pdf_path}': {e}")

    try:
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                text = page.get_text()

                # Handle potential encoding issues by normalizing the text
                if text:
                    # Replace problematic characters and normalize whitespace
                    text = text.encode("utf-8", errors="ignore").decode("utf-8")
                    text = re.sub(r"\s+", " ", text).strip()

                    if text:  # Skip pages with no extractable text
                        pages.append((page_num + 1, text))  # 1-indexed page numbers

            except Exception as e:
                print(f"  Warning: Could not extract text from page {page_num + 1}: {e}")
                continue

    finally:
        doc.close()

    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks, preferring sentence boundaries.

    Args:
        text: The text to split into chunks.
        chunk_size: Target size for each chunk in characters.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text or chunk_size <= 0:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    # Sentence-ending patterns
    sentence_end_pattern = re.compile(r"[.!?]\s+")

    while start < text_length:
        # Calculate end position for this chunk
        end = start + chunk_size

        if end >= text_length:
            # Last chunk - take everything remaining
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Try to find a sentence boundary near the end of the chunk
        # Search in the last 20% of the chunk for a good break point
        search_start = start + int(chunk_size * 0.8)
        search_region = text[search_start:end]

        # Find all sentence endings in the search region
        matches = list(sentence_end_pattern.finditer(search_region))

        if matches:
            # Use the last sentence boundary found
            last_match = matches[-1]
            actual_end = search_start + last_match.end()
        else:
            # No sentence boundary found, try to break at a space
            space_pos = text.rfind(" ", start + int(chunk_size * 0.8), end)
            if space_pos > start:
                actual_end = space_pos + 1
            else:
                # No good break point, just cut at chunk_size
                actual_end = end

        chunk = text[start:actual_end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position, accounting for overlap
        # Ensure we always make forward progress
        next_start = actual_end - overlap
        if next_start <= start:
            next_start = start + max(1, chunk_size // 2)
        
        start = next_start

    return chunks


def process_pdf(pdf_path: str, model: SentenceTransformer, collection) -> int:
    """
    Process a single PDF file and store its chunks in the vector database.

    Args:
        pdf_path: Path to the PDF file.
        model: SentenceTransformer model for generating embeddings.
        collection: ChromaDB collection to store the chunks.

    Returns:
        Total number of chunks processed and stored.

    Raises:
        RuntimeError: If the PDF cannot be processed.
    """
    pdf_path = Path(pdf_path)
    filename = pdf_path.name

    # Extract text from PDF
    pages = extract_text_from_pdf(str(pdf_path))

    if not pages:
        raise RuntimeError(f"No text could be extracted from '{filename}'")

    total_chunks = 0
    all_chunks = []
    all_ids = []
    all_metadatas = []

    for page_num, page_text in pages:
        # Chunk the page text
        chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk_idx, chunk in enumerate(chunks):
            # Create unique ID
            chunk_id = f"{filename}_p{page_num}_c{chunk_idx}"

            # Prepare metadata
            metadata = {
                "source": filename,
                "page": page_num,
                "chunk_index": chunk_idx,
            }

            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append(metadata)
            total_chunks += 1

    if all_chunks:
        # Generate embeddings for all chunks at once (more efficient)
        print(f"  Generating embeddings for {len(all_chunks)} chunks...")
        all_embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

        # Store in ChromaDB
        print(f"  Storing in database...")
        collection.add(
            ids=all_ids,
            embeddings=all_embeddings,
            documents=all_chunks,
            metadatas=all_metadatas,
        )

    return total_chunks


def main():
    """
    Main function to process all PDFs in the configured directory.

    Initializes the embedding model and vector database, then processes
    each PDF file found in PDF_DIR.
    """
    print("Initializing RAG Ingestion Pipeline...")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print(f"  PDF directory: {PDF_DIR}")
    print(f"  Database directory: {DB_DIR}")
    print(f"  Collection name: {COLLECTION_NAME}")
    print()

    # Ensure directories exist
    DB_DIR.mkdir(parents=True, exist_ok=True)

    if not PDF_DIR.exists():
        print(f"Error: PDF directory does not exist: {PDF_DIR}")
        print("Please create the directory and add PDF files to process.")
        return

    # Initialize sentence-transformers model
    print("Loading embedding model...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        print(f"Error: Failed to load embedding model: {e}")
        return

    # Initialize ChromaDB
    print("Initializing vector database...")
    try:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"Error: Failed to initialize ChromaDB: {e}")
        return

    # Find all PDF files
    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF file(s) to process.")
    print()

    # Process each PDF
    total_pdfs_processed = 0
    total_chunks = 0

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        try:
            chunks_processed = process_pdf(str(pdf_path), model, collection)
            total_chunks += chunks_processed
            total_pdfs_processed += 1
            print(f"  ✓ Extracted {chunks_processed} chunks")

        except RuntimeError as e:
            print(f"  ✗ Error: {e}")
            continue

        except Exception as e:
            print(f"  ✗ Unexpected error processing {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print()
    print("=" * 50)
    print(f"Completed! Processed {total_pdfs_processed} PDFs, {total_chunks} total chunks")
    print(f"Collection '{COLLECTION_NAME}' now contains {collection.count()} documents")


if __name__ == "__main__":
    main()