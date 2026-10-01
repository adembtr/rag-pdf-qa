# rag-pdf-qa — local, multilingual RAG over your own PDF library

Ask questions about a folder of PDFs (books, papers, notes) and get back the most relevant passages
**with source file and page number**, ready to paste into any LLM as grounded context.
Everything runs locally; it works for Turkish and English (and 50+ other languages).

```text
$ python src/query.py "What is the bias-variance trade-off?"
[Source: deep-learning-book.pdf, Page: 112]  ...relevant passage...
[Source: mml-book.pdf, Page: 278]            ...relevant passage...
→ also saved as an LLM-ready prompt in output.txt
```

## How it works

1. **Extract** — PyMuPDF reads every page; whitespace and broken encodings are normalised, empty pages skipped.
2. **Chunk** — 500-character chunks with 50-character overlap, so facts on chunk borders are not lost.
3. **Embed** — `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers, 384-d).
4. **Store** — ChromaDB persistent collection; each chunk keeps `source`, `page` and `chunk_index` metadata.
5. **Retrieve** — the question is embedded the same way and the top-k chunks are returned with citations,
   formatted as a prompt ("Relevant context … Based on the above context, please answer the question.").

All parameters live in [`src/config.py`](src/config.py) (chunk size/overlap, model, paths, collection name, `TOP_K`).

## Quick start

```bash
pip install -r requirements.txt

# 1. Put your PDFs in data/pdfs/  (they are git-ignored)
# 2. Build the vector database
python src/ingest.py

# 3. Ask questions
python src/query.py                         # interactive
python src/query.py "your question" -n 8    # single query, 8 results
python src/query.py -f question.txt -o answer_context.txt
```

## Requirements

Python 3.10+ · `pymupdf` · `sentence-transformers` · `chromadb` · `numpy` (pinned in `requirements.txt`).

## License

[MIT](LICENSE) — bring your own documents; no PDFs are distributed with this repository.

---

Built by [Adem Batur](https://github.com/adembtr) · Computer Engineering, Sakarya University
