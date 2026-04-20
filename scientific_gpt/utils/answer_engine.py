"""Generate an academic answer from a list of Paper objects using Gemini."""
from __future__ import annotations
from langchain_google_genai import ChatGoogleGenerativeAI
from .sources.base import Paper
from .reranker import filter_substantive, rerank

_PROMPT = """\
You are a rigorous scientific research assistant. Answer the question using ONLY \
the numbered sources below.

Strict citation rules:
1. Only cite a source when it DIRECTLY supports the specific claim you are making. \
   A tangentially related paper must NOT be cited — leaving a sentence uncited is \
   better than citing an unrelated source.
2. If a source listed below turns out to be irrelevant to the question, ignore it \
   entirely. Do not mention it, do not cite it.
3. Cite inline as [N], combining with [1, 3] or ranges [1–3] where appropriate.
4. Write in formal, objective academic language. No filler, no hedging phrases.
5. Do NOT add a references / bibliography section — one is appended automatically.
6. If the collected sources cannot adequately answer the question, state this \
   explicitly in one sentence and stop. Do not fabricate.
7. Length: 2–5 paragraphs depending on the question's complexity.

--- Sources (ranked by relevance, most relevant first) ---
{sources}
--- End of Sources ---

Question: {question}

Academic answer (inline [N] citations only, no reference list at the end):"""


def answer_from_papers(
    question: str,
    papers: list[Paper],
    api_key: str,
    citation_style: str = "APA",
    max_papers: int = 8,
) -> tuple[str, list[Paper]]:
    """Filter → semantically rerank → generate cited answer.

    Returns:
        (answer_text, ranked_papers) — ranked_papers is used to render the
        References section in the UI. Only papers that passed the relevance
        threshold are included.
    """
    # Stage 1: drop papers without substantive abstracts — they can't be judged
    substantive = filter_substantive(papers)

    # Stage 2: semantic re-ranking against the research question
    if substantive:
        ranked, scores = rerank(question, substantive, api_key, top_k=max_papers)
        # If the floor was too strict and everything got filtered, fall back
        if not ranked:
            ranked, scores = rerank(
                question, substantive, api_key, top_k=max_papers, apply_floor=False
            )
    else:
        # Fallback: no abstracts at all — give LLM whatever we have
        ranked = papers[:max_papers]
        scores = [0.0] * len(ranked)

    if not ranked:
        return ("The search returned no papers with sufficient metadata to answer "
                "the question. Please refine the search keywords or try other sources."), []

    # Stage 3: build context blocks
    source_blocks = []
    for i, (p, s) in enumerate(zip(ranked, scores), 1):
        authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        venue = f" — {p.venue}" if p.venue else ""
        year = f" ({p.year})" if p.year else ""
        cites = f" [cited {p.citation_count}×]" if p.citation_count else ""
        rel = f" [relevance {s:.2f}]" if s else ""
        abstract = f"\n    Abstract: {p.abstract[:600]}" if p.abstract else "\n    (no abstract)"
        source_blocks.append(
            f"[{i}] \"{p.title}\" — {authors}{year}{venue}{cites}{rel}{abstract}"
        )

    prompt = _PROMPT.format(
        sources="\n\n".join(source_blocks),
        question=question,
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.1,
    )
    return llm.invoke(prompt).content, ranked
