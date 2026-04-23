"""Find evidence for or against a claim across PDFs and/or online papers.

For each candidate source, the LLM returns one of:
  - supports:    the source directly backs the claim
  - contradicts: the source directly argues against the claim
  - unrelated:   the source doesn't speak to the claim

Output preserves the candidate ordering so PDFs stay before online papers,
matching the numbering convention used elsewhere in the app.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Literal, Optional

from .sources.base import Paper
from .reranker import filter_substantive, rerank
from .llm_backend import LLMChain

Verdict = Literal["supports", "contradicts", "unrelated"]

_VALID_VERDICTS: set[str] = {"supports", "contradicts", "unrelated"}


@dataclass
class Evidence:
    index: int                       # 1-based display index
    kind: Literal["pdf", "paper"]
    verdict: Verdict
    reason: str
    excerpt: str
    paper: Optional[Paper] = None    # set when kind == "paper"
    filename: Optional[str] = None   # set when kind == "pdf"


_JUDGE_PROMPT = """\
You are a meticulous scientific fact-checker. For the claim below, judge each \
numbered source independently.

For every source output exactly one JSON object with these fields:
- "index":   the source number (integer)
- "verdict": one of "supports", "contradicts", "unrelated"
- "reason":  ONE short sentence (<= 25 words) explaining the verdict, grounded \
in the source text

Rules:
- "supports" requires the source to directly back the specific claim. Merely \
being on the same topic is NOT support — that is "unrelated".
- "contradicts" requires the source to directly argue against the claim.
- If the source does not address the claim, use "unrelated".
- Do not fabricate content that isn't in the source.
- Output a single JSON array with one object per source, in the same order. \
No prose, no markdown fences, no trailing commentary.

Claim: {claim}

--- Sources ---
{sources}
--- End of Sources ---

JSON array:"""


def find_evidence(
    claim: str,
    chain: LLMChain,
    pdf_chunks: list | None = None,
    online_papers: list[Paper] | None = None,
    top_k_online: int = 6,
) -> list[Evidence]:
    """Collect PDF chunks + reranked online papers and judge each against the claim."""
    pdf_chunks = pdf_chunks or []
    online_papers = online_papers or []

    # Shortlist online papers by semantic similarity to the claim itself.
    ranked_online: list[Paper] = []
    if online_papers:
        substantive = filter_substantive(online_papers)
        if substantive:
            ranked_online, _ = rerank(claim, substantive, chain, top_k=top_k_online)
            if not ranked_online:
                ranked_online, _ = rerank(
                    claim, substantive, chain, top_k=top_k_online, apply_floor=False
                )
        else:
            ranked_online = online_papers[:top_k_online]

    candidates: list[Evidence] = []
    source_blocks: list[str] = []
    idx = 0

    for chunk in pdf_chunks:
        idx += 1
        filename = (
            chunk.metadata.get("source", "PDF")
            if hasattr(chunk, "metadata") else "PDF"
        )
        content = chunk.page_content.strip()
        excerpt = content[:600]
        source_blocks.append(f"[{idx}] PDF — {filename}\n    {excerpt}")
        candidates.append(Evidence(
            index=idx, kind="pdf", verdict="unrelated", reason="", excerpt=excerpt,
            filename=filename,
        ))

    for p in ranked_online:
        idx += 1
        authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        year = f" ({p.year})" if p.year else ""
        abstract = p.abstract or "(no abstract)"
        excerpt = abstract[:600]
        source_blocks.append(
            f'[{idx}] "{p.title}" — {authors}{year}\n    Abstract: {excerpt}'
        )
        candidates.append(Evidence(
            index=idx, kind="paper", verdict="unrelated", reason="", excerpt=excerpt,
            paper=p,
        ))

    if not candidates:
        return []

    prompt = _JUDGE_PROMPT.format(
        claim=claim.strip(),
        sources="\n\n".join(source_blocks),
    )
    raw = chain.invoke_chat(prompt, temperature=0.0)
    verdicts = _parse_verdicts(raw, len(candidates))

    for ev in candidates:
        v = verdicts.get(ev.index)
        if v is None:
            ev.verdict = "unrelated"
            ev.reason = "Model returned no verdict for this source."
        else:
            ev.verdict = v["verdict"]
            ev.reason = v["reason"]
    return candidates


def _parse_verdicts(raw: str, n_sources: int) -> dict[int, dict]:
    """Extract {index -> {verdict, reason}} from the LLM's JSON output.

    Tolerant of stray prose or markdown fences around the JSON array.
    """
    if not raw:
        return {}

    text = raw.strip()
    # Strip ```json ... ``` fences if the model added them.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Find the first JSON array in the text.
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return {}

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}

    if not isinstance(data, list):
        return {}

    out: dict[int, dict] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        if idx < 1 or idx > n_sources:
            continue
        verdict = str(item.get("verdict", "")).strip().lower()
        if verdict not in _VALID_VERDICTS:
            continue
        reason = str(item.get("reason", "")).strip()
        out[idx] = {"verdict": verdict, "reason": reason}
    return out
