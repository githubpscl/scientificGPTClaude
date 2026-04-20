import arxiv
from .base import Paper

SOURCE = "arXiv"


def search(query: str, limit: int = 5) -> list[Paper]:
    client = arxiv.Client()
    results = list(
        client.results(arxiv.Search(query=query, max_results=limit,
                                    sort_by=arxiv.SortCriterion.Relevance))
    )
    return [_parse(r) for r in results]


def _parse(r: arxiv.Result) -> Paper:
    return Paper(
        title=r.title,
        authors=[str(a) for a in r.authors],
        year=r.published.year if r.published else None,
        abstract=r.summary,
        url=r.entry_id,
        source=SOURCE,
        doi=r.doi,
        pdf_url=r.pdf_url,
        venue="arXiv",
        citation_count=None,
    )
