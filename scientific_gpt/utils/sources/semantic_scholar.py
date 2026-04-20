import time
import requests
from .base import Paper

_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,authors,year,abstract,citationCount,url,externalIds,venue,tldr"
SOURCE = "Semantic Scholar"


def search(query: str, limit: int = 5, retries: int = 3) -> list[Paper]:
    params = {"query": query, "limit": limit, "fields": _FIELDS}
    for attempt in range(retries):
        resp = requests.get(f"{_BASE}/paper/search", params=params, timeout=10)
        if resp.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        return [_parse(p) for p in resp.json().get("data", [])]
    raise RuntimeError("Semantic Scholar rate limit — try again shortly.")


def _parse(item: dict) -> Paper:
    eids = item.get("externalIds") or {}
    tldr_obj = item.get("tldr")
    return Paper(
        title=item.get("title") or "",
        authors=[a.get("name", "") for a in item.get("authors", [])],
        year=item.get("year"),
        abstract=item.get("abstract"),
        url=item.get("url") or "",
        source=SOURCE,
        doi=eids.get("DOI"),
        venue=item.get("venue"),
        citation_count=item.get("citationCount"),
        tldr=tldr_obj.get("text") if tldr_obj else None,
    )
