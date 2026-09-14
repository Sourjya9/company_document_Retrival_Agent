#!/usr/bin/env python3
"""
crag_core.py - Shared CRAG pipeline logic
=====================================================================
This module contains the actual CRAG pipeline (retrieval, grading,
routing, generation, logging). Both crag_pipeline.py (terminal) and
app.py (web interface) import from here, so there's exactly one
version of the pipeline logic to maintain.
"""

import os
import sys
import json
import csv
import time
import threading
from datetime import datetime
import numpy as np
import faiss
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

LLM_MODEL = "llama3.1:8b"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

HIGH_CONFIDENCE_THRESHOLD = 0.8
LOW_CONFIDENCE_THRESHOLD = 0.4

INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.json"
LOG_PATH = "query_log.csv"

HR_CONTACT = os.getenv("HR_CONTACT", "hr@yourcompany.com")
IT_CONTACT = os.getenv("IT_CONTACT", "it@yourcompany.com")

# OLLAMA_HOST lets Docker override this to reach the "ollama" container by
# name instead of localhost. Defaults to localhost for normal (non-Docker) use.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434/v1")
client = OpenAI(api_key="ollama", base_url=OLLAMA_HOST)

print("[*] Loading local embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Only one heavy LLM call is allowed to run at a time, since a single GPU
# can't truly parallelize them. This makes concurrent requests queue safely
# instead of risking a GPU memory conflict or garbled response.
llm_lock = threading.Semaphore(1)


# ---------------------------------------------------------------------------
# 2. Load the pre-built index
# ---------------------------------------------------------------------------

def load_index_and_corpus():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)):
        print(
            "[ERROR] No index found. Run 'python ingest_documents.py' first "
            "after putting your PDFs in the 'company_docs' folder."
        )
        sys.exit(1)

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunk_records = json.load(f)

    corpus = [c["text"] for c in chunk_records]
    sources = [c["source"] for c in chunk_records]
    return index, corpus, sources


def get_embedding(text: str) -> List[float]:
    clean = text.replace("\n", " ").strip()
    vec = embedder.encode(clean, normalize_embeddings=True)
    return vec.tolist()


def retrieve(index, corpus: List[str], sources: List[str], query: str, top_k: int = 3):
    q_vec = np.array([get_embedding(query)]).astype("float32")
    distances, indices = index.search(q_vec, top_k)
    return [{"text": corpus[i], "source": sources[i]} for i in indices[0] if i != -1]


# ---------------------------------------------------------------------------
# 3. LLM call wrapper - retries on transient failure, serializes GPU access
# ---------------------------------------------------------------------------

def call_llm_with_retry(**kwargs):
    """
    Wraps client.chat.completions.create with retries and a concurrency lock.
    Ollama occasionally needs a moment to load the model into GPU memory if
    it was idle - this avoids failing a request just because the first
    attempt landed during that brief warmup window. The lock ensures only
    one request uses the GPU at a time.
    """
    last_error = None
    for attempt in range(3):
        try:
            with llm_lock:
                return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_error = e
            if attempt < 2:
                time.sleep(2)
    raise last_error


# ---------------------------------------------------------------------------
# 4. Batched confidence grader
# ---------------------------------------------------------------------------

class ChunkVerdict(BaseModel):
    chunk_index: int = Field(description="Index of the chunk being graded, 0-based.")
    confidence: float = Field(description="Relevance confidence from 0.0 to 1.0.")
    reasoning: str = Field(description="One-sentence justification.")


class BatchGradeResult(BaseModel):
    verdicts: List[ChunkVerdict]


def grade_chunks_batched(query: str, chunks: List[str]) -> List[float]:
    if not chunks:
        return []

    numbered = "\n".join(f"[{i}] {c}" for i, c in enumerate(chunks))
    system_prompt = (
        "You are a strict retrieval evaluator. For EACH numbered document, assign a "
        "confidence score from 0.0 to 1.0 indicating how directly it answers the query. "
        "Do not extrapolate beyond the text. Respond with ONLY valid JSON matching this "
        'exact shape, no prose, no markdown fences:\n'
        '{"verdicts": [{"chunk_index": 0, "confidence": 0.9, "reasoning": "..."}]}\n'
        "Include one verdict per document index, in order."
    )
    user_prompt = f"Query: {query}\n\nDocuments:\n{numbered}"

    try:
        response = call_llm_with_retry(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = response.choices[0].message.content
        parsed = BatchGradeResult.model_validate_json(raw)

        scores = [0.0] * len(chunks)
        for v in parsed.verdicts:
            if 0 <= v.chunk_index < len(chunks):
                scores[v.chunk_index] = v.confidence
        return scores
    except (ValidationError, json.JSONDecodeError, Exception) as e:
        print(f"[WARN] Grading failed, defaulting all chunks to low confidence: {e}", file=sys.stderr)
        return [0.0] * len(chunks)


# ---------------------------------------------------------------------------
# 5. Generation
# ---------------------------------------------------------------------------

def synthesize_answer(query: str, context: str, source_label: str) -> str:
    system_instruction = (
        f"You are an enterprise AI assistant. Active context source: {source_label}.\n"
        "1. Answer STRICTLY from the context below.\n"
        "2. If context is insufficient, say explicitly what is missing.\n"
        "3. Never state facts not present in the context."
    )
    try:
        response = call_llm_with_retry(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] Generation failed: {e}"


# ---------------------------------------------------------------------------
# 6. Department routing (for escalation messages)
# ---------------------------------------------------------------------------

HR_KEYWORDS = [
    "pto", "vacation", "leave", "benefits", "insurance", "payroll", "salary",
    "compensation", "onboarding", "offboarding", "harassment", "discrimination",
    "performance review", "retirement", "401k", "expense", "reimbursement",
    "travel policy", "diversity", "hiring",
]
IT_KEYWORDS = [
    "password", "vpn", "wifi", "laptop", "software", "login", "access",
    "security", "mfa", "authentication", "hardware", "equipment", "network",
    "server", "database", "encryption", "it support", "computer",
]


def get_escalation_contact(query: str) -> str:
    """
    Looks at simple keyword matches to decide whether a question is more
    likely an HR or IT matter, so the escalation message points to the
    right contact instead of always listing both. Falls back to listing
    both if it's unclear.
    """
    q = query.lower()
    hr_match = any(kw in q for kw in HR_KEYWORDS)
    it_match = any(kw in q for kw in IT_KEYWORDS)

    if hr_match and not it_match:
        return f"HR ({HR_CONTACT})"
    elif it_match and not hr_match:
        return f"IT ({IT_CONTACT})"
    else:
        return f"HR ({HR_CONTACT}) or IT ({IT_CONTACT})"


# ---------------------------------------------------------------------------
# 7. Logging
# ---------------------------------------------------------------------------

def log_interaction(question: str, top_score: float, route: str, num_chunks: int, answer: str, username: str = "Anonymous"):
    """
    Append one row per question to query_log.csv. Opens directly in Excel
    or Google Sheets. This is how you spot gaps in your documentation:
    sort by confidence ascending and see what employees keep asking that
    the handbook doesn't answer well.
    """
    file_exists = os.path.exists(LOG_PATH)
    try:
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "username", "question", "top_confidence", "route", "chunks_retrieved", "answer"])
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                username,
                question,
                f"{top_score:.2f}",
                route,
                num_chunks,
                answer.replace("\n", " ")[:500],
            ])
    except Exception as e:
        print(f"[WARN] Failed to write log entry: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 8. Hybrid routing controller
# ---------------------------------------------------------------------------

def run_crag(index, corpus: List[str], sources: List[str], query: str, top_k: int = 3, verbose: bool = True, username: str = "Anonymous"):
    if verbose:
        print(f"\n=== CRAG QUERY: '{query}' ===")

    candidates = retrieve(index, corpus, sources, query, top_k=top_k)
    candidate_texts = [c["text"] for c in candidates]
    if verbose:
        print(f"[*] Retrieved {len(candidates)} candidate chunk(s).")

    scores = grade_chunks_batched(query, candidate_texts)
    if verbose:
        for c, s in zip(candidates, scores):
            print(f"    - ({s:.2f}) {c['text'][:70]}...")

    top_score = max(scores) if scores else 0.0
    relevant_chunks = [c for c, s in zip(candidates, scores) if s >= LOW_CONFIDENCE_THRESHOLD]
    relevant_texts = [c["text"] for c in relevant_chunks]
    cited_sources = sorted(set(c["source"] for c in relevant_chunks))

    if top_score >= HIGH_CONFIDENCE_THRESHOLD:
        route = "local_only"
        if verbose:
            print(f"[*] ROUTE: HIGH CONFIDENCE ({top_score:.2f}) -> local only.")
        answer = synthesize_answer(query, "\n\n".join(relevant_texts), "Internal FAISS Store")
        sources_display = cited_sources

    elif top_score >= LOW_CONFIDENCE_THRESHOLD:
        route = "partial_escalate"
        if verbose:
            print(f"[*] ROUTE: MEDIUM CONFIDENCE ({top_score:.2f}) -> partial answer + escalation note.")
        partial_answer = synthesize_answer(query, "\n\n".join(relevant_texts), "Internal FAISS Store")
        contact = get_escalation_contact(query)
        answer = (
            f"{partial_answer}\n\n"
            f"Note: I only found partial information on this in our documents. "
            f"For a complete answer, reach out to {contact}."
        )
        sources_display = cited_sources

    else:
        route = "full_escalate"
        if verbose:
            print(f"[*] ROUTE: LOW CONFIDENCE ({top_score:.2f}) -> escalating, no answer generated.")
        contact = get_escalation_contact(query)
        answer = (
            "I couldn't find anything in our company documents that answers this. "
            f"This looks like a question for {contact} \u2014 feel free to reach out directly."
        )
        sources_display = []

    log_interaction(query, top_score, route, len(candidates), answer, username)

    if verbose:
        print(f"\n[FINAL ANSWER]:\n{answer}\n")

    return {"answer": answer, "top_confidence": top_score, "route": route, "sources": sources_display}