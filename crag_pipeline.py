#!/usr/bin/env python3
"""
crag_pipeline.py - Terminal interface for the CRAG bot
=====================================================================
Run this for quick testing in the terminal. For a real employee-facing
interface, use app.py instead (a simple web page).
"""

from crag_core import load_index_and_corpus, run_crag

if __name__ == "__main__":
    print("[*] Loading pre-built document index...")
    faiss_index, KNOWLEDGE_BASE, SOURCES = load_index_and_corpus()
    print(f"[*] Loaded {len(KNOWLEDGE_BASE)} chunks from your company documents.")
    print("[*] Ask a question, or type 'quit' to exit.\n")

    while True:
        user_question = input("You: ").strip()
        if user_question.lower() in ("quit", "exit"):
            break
        if not user_question:
            continue
        run_crag(faiss_index, KNOWLEDGE_BASE, SOURCES, user_question)
