"""
EcoMoveAI - Lightweight RAG utilities.

Purpose
-------
Provide deterministic, local retrieval over project documents to enrich
LLM interpretation without external embeddings or hidden logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class RAGDocument:
    """
    Simple document container for local retrieval.
    """

    doc_id: str
    source: str
    content: str


@dataclass
class RAGIndex:
    """
    In-memory TF-IDF index for local retrieval.
    """

    documents: list[RAGDocument]
    idf: dict[str, float]
    doc_vectors: list[dict[str, float]]
    doc_norms: list[float]


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _build_tf(tokens: Iterable[str]) -> dict[str, float]:
    tf: dict[str, float] = {}
    for token in tokens:
        tf[token] = tf.get(token, 0.0) + 1.0
    return tf


def load_documents_from_dir(
    directory: Path,
    extensions: tuple[str, ...] = (".md", ".txt", ".json", ".csv"),
    max_chars_per_doc: int = 12000,
) -> list[RAGDocument]:
    """
    Load documents from a directory for retrieval.
    """

    if not directory.exists():
        raise FileNotFoundError(f"RAG directory not found: {directory}")
    documents: list[RAGDocument] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            content = path.read_text(encoding="utf-8", errors="ignore")
            if len(content) > max_chars_per_doc:
                content = content[:max_chars_per_doc]
            documents.append(
                RAGDocument(
                    doc_id=str(path.relative_to(directory)),
                    source=str(path),
                    content=content,
                )
            )
    if not documents:
        raise ValueError(f"No readable documents found in {directory}")
    return documents


def build_index(documents: list[RAGDocument]) -> RAGIndex:
    """
    Build a TF-IDF index for the given documents.
    """

    doc_freq: dict[str, int] = {}
    doc_tokens: list[list[str]] = []

    for doc in documents:
        tokens = _tokenize(doc.content)
        doc_tokens.append(tokens)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1

    total_docs = len(documents)
    idf: dict[str, float] = {}
    for token, count in doc_freq.items():
        idf[token] = math.log((total_docs + 1) / (count + 1)) + 1.0

    doc_vectors: list[dict[str, float]] = []
    doc_norms: list[float] = []
    for tokens in doc_tokens:
        tf = _build_tf(tokens)
        vector: dict[str, float] = {}
        norm_sq = 0.0
        for token, freq in tf.items():
            weight = freq * idf.get(token, 0.0)
            vector[token] = weight
            norm_sq += weight * weight
        doc_vectors.append(vector)
        doc_norms.append(math.sqrt(norm_sq) if norm_sq > 0 else 1.0)

    return RAGIndex(
        documents=documents,
        idf=idf,
        doc_vectors=doc_vectors,
        doc_norms=doc_norms,
    )


def _build_query_vector(query: str, idf: dict[str, float]) -> tuple[dict[str, float], float]:
    tokens = _tokenize(query)
    tf = _build_tf(tokens)
    vector: dict[str, float] = {}
    norm_sq = 0.0
    for token, freq in tf.items():
        weight = freq * idf.get(token, 0.0)
        vector[token] = weight
        norm_sq += weight * weight
    norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
    return vector, norm


def _extract_excerpt(content: str, query_terms: list[str], max_len: int = 400) -> str:
    lowered = content.lower()
    match_index = None
    for term in query_terms:
        idx = lowered.find(term)
        if idx != -1:
            match_index = idx
            break
    if match_index is None:
        match_index = 0
    start = max(match_index - 120, 0)
    end = min(start + max_len, len(content))
    excerpt = content[start:end].replace("\n", " ").strip()
    return excerpt


def search(index: RAGIndex, query: str, top_k: int = 3) -> list[dict[str, object]]:
    """
    Retrieve top-k documents for a query.
    """

    query_vector, query_norm = _build_query_vector(query, index.idf)
    scored: list[tuple[float, int]] = []

    for idx, doc_vector in enumerate(index.doc_vectors):
        dot = 0.0
        for token, weight in query_vector.items():
            dot += weight * doc_vector.get(token, 0.0)
        similarity = dot / (index.doc_norms[idx] * query_norm)
        scored.append((similarity, idx))

    scored.sort(reverse=True, key=lambda item: item[0])

    results: list[dict[str, object]] = []
    query_terms = _tokenize(query)
    for score, idx in scored[:top_k]:
        document = index.documents[idx]
        results.append(
            {
                "source": document.source,
                "doc_id": document.doc_id,
                "score": round(float(score), 6),
                "excerpt": _extract_excerpt(document.content, query_terms),
            }
        )
    return results
