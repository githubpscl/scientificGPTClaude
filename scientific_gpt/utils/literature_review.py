"""Generate a structured literature review from a list of academic papers."""
from __future__ import annotations

from .sources.base import Paper
from .reranker import filter_substantive, rerank
from .llm_backend import LLMChain


_PROMPT = """\
You are a scientific research assistant writing a structured literature review.

Topic: {topic}

--- Sources (numbered, ranked by relevance) ---
{sources}
--- End of Sources ---

Write a focused literature review with the following sections, in this order:

## Introduction
1–2 short paragraphs framing the topic, its scope, and why it matters. Cite \
sources where they directly motivate the framing.

## Methods and Approaches
Group the cited work by methodology, framework, or experimental approach. \
Cite [N] inline whenever you describe what a specific source did.

## Key Findings
Synthesize the substantive findings across the sources. Where sources agree, \
cite them together [1, 3]. Where they disagree, surface the disagreement \
explicitly with separate citations.

## Research Gaps
Identify gaps, unresolved questions, or contradictions visible across the \
sources. Only mention gaps that are evident from the supplied material.

## Future Directions
Suggest concrete research directions the literature points toward. Tie each \
direction to specific sources where applicable.

Strict rules:
1. Cite [N] inline for every claim derived from a specific source.
2. Do NOT fabricate findings, methods, or numbers not present in the sources.
3. Do NOT include a References / Bibliography section — one is appended \
automatically.
4. Formal academic language. No filler.
5. If the supplied sources cannot adequately cover the topic, state this \
explicitly in the Introduction and keep the review brief rather than padding.

Literature review (inline [N] citations only, no reference list at the end):"""


def _format_paper_blocks(papers: list[Paper], scores: list[float]) -> list[str]:
    blocks = []
    for i, (p, s) in enumerate(zip(papers, scores), 1):
        authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        venue = f" — {p.venue}" if p.venue else ""
        year = f" ({p.year})" if p.year else ""
        cites = f" [cited {p.citation_count}×]" if p.citation_count else ""
        rel = f" [relevance {s:.2f}]" if s else ""
        abstract = f"\n    Abstract: {p.abstract[:600]}" if p.abstract else "\n    (no abstract)"
        blocks.append(
            f"[{i}] \"{p.title}\" — {authors}{year}{venue}{cites}{rel}{abstract}"
        )
    return blocks


def _build_review_prompt(
    topic: str,
    papers: list[Paper],
    chain: LLMChain,
    max_papers: int,
) -> tuple[str | None, list[Paper]]:
    substantive = filter_substantive(papers)
    if substantive:
        ranked, scores = rerank(topic, substantive, chain, top_k=max_papers)
        if not ranked:
            ranked, scores = rerank(
                topic, substantive, chain, top_k=max_papers, apply_floor=False
            )
    else:
        ranked = papers[:max_papers]
        scores = [0.0] * len(ranked)

    if not ranked:
        return None, []

    blocks = _format_paper_blocks(ranked, scores)
    prompt = _PROMPT.format(topic=topic, sources="\n\n".join(blocks))
    return prompt, ranked


def stream_review(
    topic: str,
    papers: list[Paper],
    chain: LLMChain,
    max_papers: int = 12,
):
    """Stream a structured literature review. Returns (token_iterator, ranked).

    On no-usable-sources, the iterator yields a single fallback message and
    ranked is empty.
    """
    prompt, ranked = _build_review_prompt(topic, papers, chain, max_papers)
    if prompt is None:
        def _empty():
            yield ("The search returned no papers with sufficient metadata to "
                   "draft a literature review on this topic. Refine the keywords "
                   "or enable more sources.")
        return _empty(), []
    return chain.stream_chat(prompt, temperature=0.2), ranked
