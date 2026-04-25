"""Crossref REST API — authoritative DOI metadata, no key required."""
import requests
from .base import Paper

_BASE = "https://api.crossref.org/works"
_HEADERS = {"User-Agent": "ScientificGPT/1.0 (mailto:scientificgpt@research.app)"}
SOURCE = "Crossref"

_SELECT = ("title,author,published,DOI,URL,abstract,container-title,"
           "is-referenced-by-count,update-to,subtype")


def search(query: str, limit: int = 5) -> list[Paper]:
    params = {"query": query, "rows": limit, "select": _SELECT}
    resp = requests.get(_BASE, params=params, headers=_HEADERS, timeout=10)
    resp.raise_for_status()
    items = resp.json().get("message", {}).get("items", [])
    return [_parse(it) for it in items]


def lookup_doi(doi: str) -> Paper | None:
    """Fetch a single paper by its DOI for citation validation."""
    resp = requests.get(f"{_BASE}/{doi}", headers=_HEADERS, timeout=10)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return _parse(resp.json().get("message", {}))


def _parse(item: dict) -> Paper:
    titles = item.get("title") or []
    title = titles[0] if titles else ""

    authors = []
    for a in item.get("author") or []:
        given = a.get("given") or ""
        family = a.get("family") or ""
        name = f"{given} {family}".strip()
        if name:
            authors.append(name)

    year = None
    pub = item.get("published") or item.get("published-print") or item.get("published-online")
    if pub:
        dp = pub.get("date-parts", [[]])
        if dp and dp[0]:
            try:
                year = int(dp[0][0])
            except (ValueError, TypeError):
                pass

    containers = item.get("container-title") or []
    venue = containers[0] if containers else None

    doi = item.get("DOI")
    url = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")

    # Strip HTML from abstract
    abstract_raw = item.get("abstract") or ""
    abstract = _strip_jats(abstract_raw) or None

    # Retraction detection: Crossref lists `update-to` entries with a `type`
    # field — "retraction" / "withdrawal" indicates the paper was retracted.
    is_retracted = False
    for upd in item.get("update-to") or []:
        utype = (upd.get("type") or "").lower()
        if "retract" in utype or "withdraw" in utype:
            is_retracted = True
            break
    if (item.get("subtype") or "").lower() in ("retraction", "withdrawal"):
        is_retracted = True

    return Paper(
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        url=url,
        source=SOURCE,
        doi=doi,
        venue=venue,
        citation_count=item.get("is-referenced-by-count"),
        is_retracted=is_retracted,
    )


def _strip_jats(text: str) -> str:
    """Remove simple JATS/XML tags from Crossref abstract strings."""
    import re
    return re.sub(r"<[^>]+>", "", text).strip()
