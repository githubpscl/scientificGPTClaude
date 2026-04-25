"""Unified parallel search across all academic sources."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Literal
import time

import requests

from .sources.base import Paper
from .sources import (
    semantic_scholar,
    openalex,
    arxiv_source,
    pubmed_source,
    crossref,
    core_source,
)

SortKey = Literal["relevance", "year", "citations"]

# Transient errors retried inside _fetch (network blips, server-side hiccups).
_TRANSIENT_EXC = (requests.exceptions.Timeout, requests.exceptions.ConnectionError)
_TRANSIENT_MARKERS = ("500", "502", "503", "504", "timeout", "temporarily")

# Registry: name → (search_fn, always_available)
_SOURCES: dict[str, tuple[Callable, bool]] = {
    "Semantic Scholar": (semantic_scholar.search, True),
    "OpenAlex":         (openalex.search,          True),
    "arXiv":            (arxiv_source.search,       True),
    "PubMed":           (pubmed_source.search,      True),
    "Crossref":         (crossref.search,           True),
    "CORE":             (core_source.search,        False),  # needs API key
}

# Colour badge for each source (Streamlit markdown)
SOURCE_COLORS: dict[str, str] = {
    "Semantic Scholar": "#1a73e8",
    "OpenAlex":         "#0f9d58",
    "arXiv":            "#b7091c",
    "PubMed":           "#005596",
    "Crossref":         "#f4a100",
    "CORE":             "#5e35b1",
}


def available_sources() -> list[str]:
    return list(_SOURCES.keys())


def is_source_available(name: str) -> bool:
    """Returns False for sources that need an unconfigured API key."""
    _, always = _SOURCES[name]
    if not always and name == "CORE":
        return core_source.is_available()
    return True


def search(
    query: str,
    sources: list[str],
    limit_per_source: int = 5,
    min_year: int | None = None,
    language: str | None = None,
    open_access_only: bool = False,
    min_citations: int | None = None,
    exclude_retracted: bool = True,
    sort_by: SortKey = "relevance",
) -> tuple[list[Paper], dict[str, str]]:
    """Run parallel search across selected sources.

    Filters:
        min_year:           drop papers older than this year (strict — unknown year dropped).
        language:           ISO 639-1 code; drop papers whose language is known and differs
                            (permissive — unknown language kept).
        open_access_only:   drop papers explicitly marked non-OA. Unknown is kept.
        min_citations:      drop papers below threshold (papers without a count are kept).
        exclude_retracted:  drop papers flagged as retracted by Crossref.
        sort_by:            "relevance" (source order), "year" desc, or "citations" desc.

    Returns:
        papers: deduplicated, filtered, sorted list.
        errors: {source_name: error_message} for any failed source.
    """
    if not sources:
        return [], {}

    results: list[Paper] = []
    errors: dict[str, str] = {}

    def _fetch(name: str) -> list[Paper]:
        fn, _ = _SOURCES[name]
        last_exc: BaseException | None = None
        for attempt in range(3):
            try:
                return fn(query, limit_per_source)
            except _TRANSIENT_EXC as e:
                last_exc = e
            except Exception as e:
                msg = str(e).lower()
                if any(m in msg for m in _TRANSIENT_MARKERS):
                    last_exc = e
                else:
                    raise
            time.sleep(0.6 * (2 ** attempt))   # 0.6 s, 1.2 s, 2.4 s
        assert last_exc is not None
        raise last_exc

    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        future_map = {pool.submit(_fetch, name): name for name in sources}
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                results.extend(future.result())
            except Exception as e:
                # Shorten HTTP error messages for display in the UI
                msg = str(e)
                if "400" in msg:
                    errors[name] = "Bad request — query may contain unsupported characters."
                elif "429" in msg or "rate limit" in msg.lower():
                    errors[name] = "Rate limit reached — try again in a few seconds."
                elif "500" in msg or "502" in msg or "503" in msg:
                    errors[name] = "Source temporarily unavailable (server error)."
                elif "timeout" in msg.lower():
                    errors[name] = "Request timed out."
                else:
                    errors[name] = _scrub_secrets(msg)

    filtered = _apply_filters(
        results,
        min_year=min_year,
        language=language,
        open_access_only=open_access_only,
        min_citations=min_citations,
        exclude_retracted=exclude_retracted,
    )
    deduped = _deduplicate(filtered)
    return _sort(deduped, sort_by), errors


def _apply_filters(
    papers: list[Paper],
    *,
    min_year: int | None,
    language: str | None,
    open_access_only: bool,
    min_citations: int | None,
    exclude_retracted: bool,
) -> list[Paper]:
    out = []
    lang = (language or "").strip().lower() or None
    for p in papers:
        if min_year is not None:
            if p.year is None or p.year < min_year:
                continue
        if lang is not None:
            p_lang = (p.language or "").strip().lower()
            if p_lang and p_lang != lang:
                continue
        if open_access_only and p.is_open_access is False:
            continue
        if min_citations is not None and p.citation_count is not None:
            if p.citation_count < min_citations:
                continue
        if exclude_retracted and p.is_retracted:
            continue
        out.append(p)
    return out


def _sort(papers: list[Paper], key: SortKey) -> list[Paper]:
    if key == "year":
        return sorted(papers, key=lambda p: (p.year or 0), reverse=True)
    if key == "citations":
        return sorted(papers, key=lambda p: (p.citation_count or 0), reverse=True)
    return papers  # relevance = original order


def _scrub_secrets(msg: str) -> str:
    """Best-effort removal of long alphanumeric tokens that look like API keys."""
    from .secrets_filter import scrub
    return scrub(msg)


def _deduplicate(papers: list[Paper]) -> list[Paper]:
    seen: set[str] = set()
    unique: list[Paper] = []
    for p in papers:
        key = p.dedup_key()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique
