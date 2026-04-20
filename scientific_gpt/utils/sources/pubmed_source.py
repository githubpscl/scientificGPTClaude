"""PubMed via NCBI E-utilities (no API key required; email used for polite pool)."""
import xml.etree.ElementTree as ET
import requests
from .base import Paper

_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_EMAIL = "scientificgpt@research.app"
SOURCE = "PubMed"


def search(query: str, limit: int = 5) -> list[Paper]:
    # Step 1: get PMIDs
    params = {"db": "pubmed", "term": query, "retmax": limit,
              "retmode": "json", "email": _EMAIL}
    resp = requests.get(_ESEARCH, params=params, timeout=10)
    resp.raise_for_status()
    ids = resp.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    # Step 2: fetch abstracts as XML
    params2 = {"db": "pubmed", "id": ",".join(ids), "rettype": "abstract",
               "retmode": "xml", "email": _EMAIL}
    resp2 = requests.get(_EFETCH, params=params2, timeout=15)
    resp2.raise_for_status()

    return _parse_xml(resp2.text)


def _parse_xml(xml_text: str) -> list[Paper]:
    papers = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    for article in root.findall(".//PubmedArticle"):
        citation = article.find("MedlineCitation")
        if citation is None:
            continue
        art = citation.find("Article")
        if art is None:
            continue

        title_el = art.find("ArticleTitle")
        title = "".join(title_el.itertext()) if title_el is not None else ""

        # Abstract
        abstract_parts = [
            "".join(t.itertext())
            for t in art.findall(".//AbstractText")
        ]
        abstract = " ".join(abstract_parts) or None

        # Authors
        authors = []
        for au in art.findall(".//Author"):
            last = au.findtext("LastName") or ""
            fore = au.findtext("ForeName") or ""
            name = f"{fore} {last}".strip()
            if name:
                authors.append(name)

        # Year
        pub_date = art.find(".//PubDate")
        year = None
        if pub_date is not None:
            year_el = pub_date.find("Year")
            if year_el is not None and year_el.text:
                try:
                    year = int(year_el.text)
                except ValueError:
                    pass

        # Journal
        journal_el = art.find("Journal")
        venue = journal_el.findtext("Title") if journal_el is not None else None

        # PMID
        pmid_el = citation.find("PMID")
        pmid = pmid_el.text if pmid_el is not None else None
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

        # DOI
        doi = None
        for eid in article.findall(".//ArticleId"):
            if eid.get("IdType") == "doi":
                doi = eid.text
                break

        papers.append(Paper(
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            url=url,
            source=SOURCE,
            doi=doi,
            pdf_url=None,
            venue=venue,
            citation_count=None,
        ))
    return papers
