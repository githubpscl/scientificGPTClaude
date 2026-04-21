"""Semantic re-ranking of papers by cosine similarity to the research question.

Why: Keyword search returns many tangential results. Embeddings let us score
each paper's abstract against the actual research question and keep only the
most relevant ones before feeding them to the answer-generation LLM.
"""
from __future__ import annotations
import numpy as np
from .sources.base import Paper
from .llm_providers import LLMConfig, get_embeddings

_RELEVANCE_FLOOR = 0.55
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
    config: LLMConfig,
    top_k: int = 8,
    apply_floor: bool = True,
) -> tuple[list[Paper], list[float]]:
    """Return (top_k papers, relevance scores) sorted by cosine similarity.

    If the provider has no embeddings (e.g. Anthropic), returns the first
    top_k papers unchanged with zero scores — the answer LLM will still get
    the full filtered list, just without semantic reordering.
    """
    if not papers:
        return [], []
    if len(papers) == 1:
        return papers, [1.0]

    emb = get_embeddings(config)
    if emb is None:
        # Provider has no embeddings — skip semantic rerank, preserve source order.
        return papers[:top_k], [0.0] * min(top_k, len(papers))

    q_vec = np.asarray(emb.embed_query(question), dtype=np.float32)

    paper_texts = [
        f"{p.title}. {p.abstract or ''}".strip()
        for p in papers
    ]
    p_vecs = np.asarray(emb.embed_documents(paper_texts), dtype=np.float32)

    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-12)
    p_norms = p_vecs / (np.linalg.norm(p_vecs, axis=1, keepdims=True) + 1e-12)
    scores = (p_norms @ q_norm).tolist()

    ranked = sorted(zip(papers, scores), key=lambda x: x[1], reverse=True)

    if apply_floor:
        ranked = [(p, s) for p, s in ranked if s >= _RELEVANCE_FLOOR]

    top = ranked[:top_k]
    return [p for p, _ in top], [round(s, 3) for _, s in top]
