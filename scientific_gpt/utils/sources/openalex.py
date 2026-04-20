import requests
from .base import Paper

_BASE = "https://api.openalex.org/works"
_HEADERS = {"User-Agent": "ScientificGPT/1.0 (mailto:scientificgpt@research.app)"}
SOURCE = "OpenAlex"

_SELECT = (
    "title,authorships,publication_year,abstract_inverted_index,"
    "doi,primary_location,cited_by_count,open_access,best_oa_location,host_venue"
)


def search(query: str, limit: int = 5) -> list[Paper]:
    params = {"search": query, "per-page": min(limit, 25), "select": _SELECT}
    resp = requests.get(_BASE, params=params, headers=_HEADERS, timeout=10)
    resp.raise_for_status()
    return [_parse(w) for w in resp.json().get("results", [])]


def _parse(w: dict) -> Paper:
    doi = w.get("doi") or ""
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    elif doi.startswith("http://doi.org/"):
        doi = doi[len("http://doi.org/"):]

    # Best open-access PDF
    oa = w.get("best_oa_location") or w.get("open_access") or {}
    pdf_url = oa.get("pdf_url") or oa.get("landing_page_url")

    # Landing page
    primary = w.get("primary_location") or {}
    url = primary.get("landing_page_url") or (f"https://doi.org/{doi}" if doi else "")

    # Venue
    venue_obj = w.get("host_venue") or {}
    venue = venue_obj.get("display_name") or primary.get("source", {}).get("display_name", "")

    return Paper(
        title=w.get("title") or "",
        authors=[
            a["author"]["display_name"]
            for a in w.get("authorships", [])
            if a.get("author")
        ],
        year=w.get("publication_year"),
        abstract=_reconstruct_abstract(w.get("abstract_inverted_index")),
        url=url,
        source=SOURCE,
        doi=doi or None,
        pdf_url=pdf_url,
        venue=venue or None,
        citation_count=w.get("cited_by_count"),
    )


def _reconstruct_abstract(inv: dict | None) -> str | None:
    if not inv:
        return None
    positions: dict[int, str] = {}
    for word, pos_list in inv.items():
        for pos in pos_list:
            positions[pos] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))
