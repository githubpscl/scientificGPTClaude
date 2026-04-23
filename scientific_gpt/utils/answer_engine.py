"""Generate an academic answer from papers and/or PDF chunks."""
from __future__ import annotations
from dataclasses import dataclass
from .sources.base import Paper
from .reranker import filter_substantive, rerank
from .llm_backend import LLMChain

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


_MIXED_PROMPT = """\
You are a rigorous scientific research assistant. Answer the question using ONLY \
the numbered sources below. Sources include both excerpts from the user's \
uploaded PDFs (marked "PDF") and external academic papers retrieved from \
online databases.

Strict citation rules:
1. Only cite a source when it DIRECTLY supports the specific claim you are making. \
   Prefer the PDF sources when they cover the claim; use external papers to \
   complement or contrast.
2. If a source turns out to be irrelevant to the question, ignore it entirely.
3. Cite inline as [N], combining with [1, 3] or ranges [1–3] where appropriate.
4. Write in formal, objective academic language.
5. Do NOT add a references / bibliography section — one is appended automatically.
6. If the sources cannot adequately answer the question, state this explicitly \
   and stop. Do not fabricate.
7. Length: 2–5 paragraphs.

--- Sources ---
{sources}
--- End of Sources ---

Question: {question}

Academic answer (inline [N] citations only, no reference list at the end):"""


@dataclass
class PdfRef:
    """Rendering info for a PDF chunk used as a citation."""
    index: int
    filename: str
    preview: str


def answer_from_papers(
    question: str,
    papers: list[Paper],
    chain: LLMChain,
    citation_style: str = "APA",
    max_papers: int = 8,
) -> tuple[str, list[Paper]]:
    """Filter → semantically rerank → generate cited answer from online papers only."""
    substantive = filter_substantive(papers)

    if substantive:
        ranked, scores = rerank(question, substantive, chain, top_k=max_papers)
        if not ranked:
            ranked, scores = rerank(
                question, substantive, chain, top_k=max_papers, apply_floor=False
            )
    else:
        ranked = papers[:max_papers]
        scores = [0.0] * len(ranked)

    if not ranked:
        return ("The search returned no papers with sufficient metadata to answer "
                "the question. Please refine the search keywords or try other sources."), []

    source_blocks = _format_paper_blocks(ranked, scores, start_index=1)

    prompt = _PROMPT.format(
        sources="\n\n".join(source_blocks),
        question=question,
    )

    answer = chain.invoke_chat(prompt, temperature=0.1)
    return answer, ranked


def answer_from_mixed(
    question: str,
    pdf_chunks: list,
    online_papers: list[Paper],
    chain: LLMChain,
    citation_style: str = "APA",
    max_online: int = 6,
) -> tuple[str, list[PdfRef], list[Paper]]:
    """Generate a cited answer combining PDF chunks and online papers.

    Returns (answer_text, pdf_refs, ranked_online_papers). Citation numbers
    run through PDFs first, then online papers.
    """
    substantive = filter_substantive(online_papers)
    if substantive:
        ranked, scores = rerank(question, substantive, chain, top_k=max_online)
        if not ranked:
            ranked, scores = rerank(
                question, substantive, chain, top_k=max_online, apply_floor=False
            )
    else:
        ranked = online_papers[:max_online]
        scores = [0.0] * len(ranked)

    source_blocks: list[str] = []
    pdf_refs: list[PdfRef] = []
    idx = 0

    for chunk in pdf_chunks:
        idx += 1
        filename = chunk.metadata.get("source", "PDF") if hasattr(chunk, "metadata") else "PDF"
        content = chunk.page_content.strip()
        source_blocks.append(f"[{idx}] PDF — {filename}\n    {content[:600]}")
        pdf_refs.append(PdfRef(index=idx, filename=filename, preview=content[:300]))

    source_blocks.extend(
        _format_paper_blocks(ranked, scores, start_index=idx + 1)
    )

    if not source_blocks:
        return ("No PDF chunks and no online papers were available. Upload PDFs "
                "or refine the search keywords."), [], []

    prompt = _MIXED_PROMPT.format(
        sources="\n\n".join(source_blocks),
        question=question,
    )

    answer = chain.invoke_chat(prompt, temperature=0.1)
    return answer, pdf_refs, ranked


def _format_paper_blocks(papers: list[Paper], scores: list[float], start_index: int) -> list[str]:
    blocks = []
    for offset, (p, s) in enumerate(zip(papers, scores)):
        i = start_index + offset
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
