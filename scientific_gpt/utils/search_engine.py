"""Unified parallel search across all academic sources."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
import traceback

from .sources.base import Paper
from .sources import (
    semantic_scholar,
    openalex,
    arxiv_source,
    pubmed_source,
    crossref,
    core_source,
)

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
) -> tuple[list[Paper], dict[str, str]]:
    """Run parallel search across selected sources.

    Returns:
        papers: deduplicated list ordered by source then relevance.
        errors: {source_name: error_message} for any failed source.
    """
    results: list[Paper] = []
    errors: dict[str, str] = {}

    def _fetch(name: str) -> list[Paper]:
        fn, _ = _SOURCES[name]
        return fn(query, limit_per_source)

    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        future_map = {pool.submit(_fetch, name): name for name in sources}
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                results.extend(future.result())
            except Exception as e:
                errors[name] = str(e)

    return _deduplicate(results), errors


def _deduplicate(papers: list[Paper]) -> list[Paper]:
    seen: set[str] = set()
    unique: list[Paper] = []
    for p in papers:
        key = p.dedup_key()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique
