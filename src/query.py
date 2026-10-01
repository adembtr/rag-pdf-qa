"""
Query Script for RAG System.

This script searches the vector database for relevant text chunks
based on a user's question and formats the results for display
or for use as context in an LLM prompt.

Usage:
    python query.py                       # Interactive mode
    python query.py "your question"       # Single query mode
    python query.py --file query.txt      # Read query from file, save to output.txt
    python query.py -f query.txt          # Short form
"""

import argparse
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import COLLECTION_NAME, DB_DIR, EMBEDDING_MODEL, TOP_K


def load_database() -> tuple[SentenceTransformer, chromadb.Collection]:
    """
    Load the embedding model and connect to the vector database.
    """
    if not DB_DIR.exists():
        raise FileNotFoundError(
            f"Database directory not found: {DB_DIR}\n"
            "Please run 'python ingest.py' first to process PDF files."
        )

    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' not found.\n"
            "Please run 'python ingest.py' first to process PDF files."
        )

    if collection.count() == 0:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' is empty.\n"
            "Please run 'python ingest.py' first to process PDF files."
        )

    return model, collection


def search(
    query: str,
    model: SentenceTransformer,
    collection: chromadb.Collection,
    n_results: int = TOP_K,
) -> list[dict]:
    """
    Search the vector database for chunks relevant to the query.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    formatted_results = []

    if results["documents"] and results["documents"][0]:
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, metadata, distance in zip(documents, metadatas, distances):
            formatted_results.append(
                {
                    "text": doc,
                    "source": metadata.get("source", "Unknown"),
                    "page": metadata.get("page", 0),
                    "score": distance,
                    "chunk_index": metadata.get("chunk_index", 0),
                }
            )

    return formatted_results


def format_results(results: list[dict]) -> str:
    """
    Format search results for human-readable display.
    """
    if not results:
        return "No results found."

    lines = []
    lines.append(f"\nFound {len(results)} relevant chunk(s):\n")

    for i, result in enumerate(results, 1):
        similarity = max(0, 1 - result["score"] / 2)

        lines.append("═" * 50)
        lines.append(
            f"[{i}] Score: {similarity:.2f} | "
            f"Source: {result['source']} | "
            f"Page: {result['page']}"
        )
        lines.append("─" * 50)

        text = result["text"]
        if len(text) > 500:
            text = text[:500] + "..."

        lines.append(f'"{text}"')
        lines.append("")

    return "\n".join(lines)


def format_for_llm(query: str, results: list[dict]) -> str:
    """
    Format search results as context for an LLM prompt.
    """
    lines = []

    lines.append(f"Question: {query}")
    lines.append("")
    lines.append("Relevant context from documents:")
    lines.append("")

    if not results:
        lines.append("[No relevant documents found]")
    else:
        for result in results:
            lines.append(f"[Source: {result['source']}, Page: {result['page']}]")
            lines.append(result["text"])
            lines.append("")

    lines.append("Based on the above context, please answer the question.")

    return "\n".join(lines)


def read_query_from_file(file_path: str) -> str:
    """
    Read query text from a file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Query file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        query = f.read().strip()

    if not query:
        raise ValueError(f"Query file is empty: {file_path}")

    return query


def save_output(content: str, output_path: str):
    """
    Save content to a file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def run_query(query: str, model, collection, output_file: str = None):
    """
    Execute a query and save results to file or display.
    """
    try:
        results = search(query, model, collection)
        
        # Build output content
        output_lines = []
        output_lines.append("RAG Query Results")
        output_lines.append("=" * 50)
        output_lines.append("")
        
        # Truncate query display if too long
        display_query = query[:200] + "..." if len(query) > 200 else query
        output_lines.append(f"Query: {display_query}")
        output_lines.append("")
        
        output_lines.append(format_results(results))
        output_lines.append("")
        output_lines.append("=" * 50)
        output_lines.append("LLM-Ready Prompt:")
        output_lines.append("=" * 50)
        output_lines.append("")
        output_lines.append(format_for_llm(query, results))
        
        full_output = "\n".join(output_lines)
        
        if output_file:
            save_output(full_output, output_file)
            print(f"✓ Results saved to: {output_file}")
        else:
            print(full_output)
            
    except ValueError as e:
        print(f"Error: {e}")


def main():
    """
    Main function for querying the RAG database.
    """
    parser = argparse.ArgumentParser(
        description="Query the RAG database for relevant text chunks.",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="The query text (optional if using --file)",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to a file containing the query text",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.txt",
        help="Output file path (default: output.txt)",
    )
    parser.add_argument(
        "-n",
        "--num-results",
        type=int,
        default=TOP_K,
        help=f"Number of results to return (default: {TOP_K})",
    )

    args = parser.parse_args()

    print("RAG Query System")
    print("=" * 50)

    # Load database
    try:
        print("Loading database...")
        model, collection = load_database()
        print(f"Connected to collection '{COLLECTION_NAME}' ({collection.count()} documents)")
        print()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Unexpected error loading database: {e}")
        return

    # Determine query source
    if args.file:
        # File mode: read query from file, save to output file
        try:
            query = read_query_from_file(args.file)
            print(f"Reading query from: {args.file}")
            print(f"Query length: {len(query)} characters")
            print(f"Output will be saved to: {args.output}")
            print()
            print("Searching...")
            run_query(query, model, collection, output_file=args.output)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            return

    elif args.query:
        # Single query mode from command line
        query = " ".join(args.query)
        run_query(query, model, collection, output_file=args.output)

    else:
        # Interactive mode (no file output)
        print("Interactive Mode")
        print("Enter your question (or 'quit' to exit):")
        print("Tip: Use 'python query.py -f query.txt' for long queries")
        print()

        while True:
            try:
                query = input(">>> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not query:
                print("Please enter a question.\n")
                continue

            try:
                results = search(query, model, collection)
                print(format_results(results))
                print("=" * 50)
                print("LLM-Ready Prompt (copy and paste to Claude):")
                print("=" * 50)
                print(format_for_llm(query, results))
                print()
                print("-" * 50)
                print("Enter your next question (or 'quit' to exit):")
                print()

            except ValueError as e:
                print(f"Error: {e}\n")
            except Exception as e:
                print(f"Search error: {e}\n")


if __name__ == "__main__":
    main()