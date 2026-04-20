"""Semantic re-ranking of papers by cosine similarity to the research question.

Why: Keyword search returns many tangential results. Gemini embeddings let us
score each paper's abstract against the actual research question and keep only
the most relevant ones before feeding them to the answer-generation LLM.
"""
from __future__ import annotations
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from .sources.base import Paper

# Papers below this relevance score are dropped regardless of rank
_RELEVANCE_FLOOR = 0.55
# Drop abstracts shorter than this — not substantive enough to assess
_MIN_ABSTRACT_LEN = 120


def filter_substantive(papers: list[Paper]) -> list[Paper]:
    """Keep only papers with a meaningful abstract."""
    return [
        p for p in papers
        if p.abstract and len(p.abstract.strip()) >= _MIN_ABSTRACT_LEN
    ]


def rerank(
    question: str,
    papers: list[Paper],
    api_key: str,
    top_k: int = 8,
    apply_floor: bool = True,
) -> tuple[list[Paper], list[float]]:
    """Return (top_k papers, their relevance scores) sorted by cosine similarity.

    Papers scoring below _RELEVANCE_FLOOR are dropped if apply_floor=True — this
    prevents the answer LLM from citing tangentially related sources.
    """
    if not papers:
        return [], []
    if len(papers) == 1:
        return papers, [1.0]

    emb = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", google_api_key=api_key
    )

    # Task-type "retrieval_query" + "retrieval_document" would be ideal but
    # the LangChain wrapper doesn't expose it cleanly; default works well.
    q_vec = np.asarray(emb.embed_query(question), dtype=np.float32)

    # Combine title + abstract for richer representation
    paper_texts = [
        f"{p.title}. {p.abstract or ''}".strip()
        for p in papers
    ]
    p_vecs = np.asarray(emb.embed_documents(paper_texts), dtype=np.float32)

    # Cosine similarity (both vectors L2-normalised)
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-12)
    p_norms = p_vecs / (np.linalg.norm(p_vecs, axis=1, keepdims=True) + 1e-12)
    scores = (p_norms @ q_norm).tolist()

    ranked = sorted(zip(papers, scores), key=lambda x: x[1], reverse=True)

    if apply_floor:
        ranked = [(p, s) for p, s in ranked if s >= _RELEVANCE_FLOOR]

    top = ranked[:top_k]
    return [p for p, _ in top], [round(s, 3) for _, s in top]
