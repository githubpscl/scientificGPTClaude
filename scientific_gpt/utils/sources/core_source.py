"""CORE API — world's largest open-access aggregator with full-text links.

Requires a free API key from https://core.ac.uk/services/api
Set CORE_API_KEY in your .env file.
"""
import os
import requests
from .base import Paper

_BASE = "https://api.core.ac.uk/v3"
SOURCE = "CORE"


def is_available() -> bool:
    return bool(os.getenv("CORE_API_KEY", "").strip())


def search(query: str, limit: int = 5) -> list[Paper]:
    api_key = os.getenv("CORE_API_KEY", "").strip()
    if not api_key:
        return []
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"q": query, "limit": limit}
    resp = requests.get(f"{_BASE}/search/works", params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    return [_parse(w) for w in resp.json().get("results", [])]


def _parse(w: dict) -> Paper:
    doi = w.get("doi")
    return Paper(
        title=w.get("title") or "",
        authors=[a.get("name", "") for a in (w.get("authors") or [])],
        year=w.get("yearPublished"),
        abstract=w.get("abstract"),
        url=w.get("sourceFulltextUrls", [None])[0] or w.get("downloadUrl") or "",
        source=SOURCE,
        doi=doi,
        pdf_url=w.get("downloadUrl"),
        venue=w.get("journals", [{}])[0].get("title") if w.get("journals") else None,
        citation_count=None,
    )
