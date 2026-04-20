import time
import requests

BASE_URL = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,authors,year,abstract,citationCount,url,externalIds,venue"


def search_papers(query: str, limit: int = 5, retries: int = 3) -> list[dict]:
    params = {"query": query, "limit": limit, "fields": _FIELDS}
    for attempt in range(retries):
        resp = requests.get(f"{BASE_URL}/paper/search", params=params, timeout=10)
        if resp.status_code == 429:
            wait = 2 ** attempt  # 1s, 2s, 4s
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json().get("data", [])
    raise RuntimeError("Semantic Scholar API rate limit exceeded — please try again in a moment.")


# ── Citation formatters ────────────────────────────────────────────────────────

def _author_list(authors: list[dict]) -> list[str]:
    return [a.get("name", "Unknown") for a in authors]


def format_citation(paper: dict, style: str) -> str:
    title = paper.get("title") or "Unknown Title"
    authors = _author_list(paper.get("authors", []))
    year = paper.get("year") or "n.d."
    url = paper.get("url") or ""
    venue = paper.get("venue") or ""

    if style == "APA":
        a_str = _apa_authors(authors)
        venue_part = f" *{venue}*." if venue else ""
        return f"{a_str} ({year}). {title}.{venue_part} {url}".strip()

    if style == "MLA":
        first = _mla_first_author(authors)
        rest = ", et al." if len(authors) > 1 else ""
        venue_part = f" *{venue}*," if venue else ""
        return f'{first}{rest}. "{title}."{venue_part} {year}. {url}'.strip()

    if style == "IEEE":
        a_str = _ieee_authors(authors)
        venue_part = f", in *{venue}*" if venue else ""
        return f'{a_str}, "{title}"{venue_part}, {year}. [Online]. Available: {url}'.strip()

    return f"{title} ({year})"


def _apa_authors(names: list[str]) -> str:
    formatted = []
    for n in names[:7]:
        parts = n.split()
        if len(parts) >= 2:
            last = parts[-1]
            initials = ". ".join(p[0].upper() for p in parts[:-1]) + "."
            formatted.append(f"{last}, {initials}")
        else:
            formatted.append(n)
    if len(names) > 7:
        formatted = formatted[:6] + ["... " + formatted[-1]]
    return ", ".join(formatted) if formatted else "Unknown"


def _mla_first_author(names: list[str]) -> str:
    if not names:
        return "Unknown"
    parts = names[0].split()
    if len(parts) >= 2:
        return f"{parts[-1]}, {' '.join(parts[:-1])}"
    return names[0]


def _ieee_authors(names: list[str]) -> str:
    result = []
    for n in names[:3]:
        parts = n.split()
        if len(parts) >= 2:
            initials = ". ".join(p[0].upper() for p in parts[:-1])
            result.append(f"{initials}. {parts[-1]}")
        else:
            result.append(n)
    suffix = " et al." if len(names) > 3 else ""
    return ", ".join(result) + suffix
