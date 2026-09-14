#!/usr/bin/env python3
"""
Document Ingestion Script - run this ONCE (or whenever your PDFs change)
=========================================================================
Reads every PDF in ./company_docs/, splits each into ~300-word chunks,
embeds them locally, and saves the result to disk so the main bot
(crag_pipeline.py) can load it instantly instead of re-processing PDFs
on every startup.

Usage:
  1. Create a folder called "company_docs" next to this script.
  2. Drop your HR/IT/policy PDFs into it.
  3. Run: python ingest_documents.py
  4. This creates faiss_index.bin and chunks.json - the bot reads these.

Re-run this script any time you add, remove, or update a PDF.
"""

import os
import json
import glob
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DOCS_FOLDER = "company_docs"
INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.json"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
WORDS_PER_CHUNK = 300
OVERLAP_WORDS = 50  # small overlap so answers don't get cut off at chunk boundaries


def extract_text_from_pdf(path: str) -> str:
    """Pull all text out of a PDF, page by page."""
    reader = PdfReader(path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)


def chunk_text(text: str, source_name: str) -> list[dict]:
    """
    Split long text into overlapping word-based chunks.
    Overlap prevents a fact from being awkwardly split across two chunks
    with neither half being enough to answer a question about it.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + WORDS_PER_CHUNK
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append({
                "text": " ".join(chunk_words),
                "source": source_name,
            })
        start += WORDS_PER_CHUNK - OVERLAP_WORDS
    return chunks


def main():
    if not os.path.isdir(DOCS_FOLDER):
        print(f"[ERROR] Folder '{DOCS_FOLDER}' not found. Create it and add PDFs first.")
        return

    pdf_paths = glob.glob(os.path.join(DOCS_FOLDER, "*.pdf"))
    if not pdf_paths:
        print(f"[ERROR] No PDF files found in '{DOCS_FOLDER}'.")
        return

    print(f"[*] Found {len(pdf_paths)} PDF(s). Extracting and chunking...")
    all_chunks = []
    for path in pdf_paths:
        name = os.path.basename(path)
        try:
            text = extract_text_from_pdf(path)
            if not text.strip():
                print(f"    [WARN] '{name}' produced no extractable text (scanned image PDF?). Skipping.")
                continue
            doc_chunks = chunk_text(text, name)
            all_chunks.extend(doc_chunks)
            print(f"    - {name}: {len(doc_chunks)} chunk(s)")
        except Exception as e:
            print(f"    [WARN] Failed to read '{name}': {e}")

    if not all_chunks:
        print("[ERROR] No usable text extracted from any PDF. Aborting.")
        return

    print(f"[*] Total chunks: {len(all_chunks)}. Embedding locally (this may take a moment)...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("[*] Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved '{INDEX_PATH}' and '{CHUNKS_PATH}'.")
    print(f"       {len(all_chunks)} chunks from {len(pdf_paths)} PDF(s) are ready for the bot.")


if __name__ == "__main__":
    main()
