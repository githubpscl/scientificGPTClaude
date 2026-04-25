from __future__ import annotations
from dataclasses import dataclass, field

# Supported citation styles — kept as a constant so the UI can import it.
CITATION_STYLES: list[str] = [
    "APA",        # Psychology, Social Sciences
    "MLA",        # Humanities
    "IEEE",       # Engineering, CS
    "Chicago",    # Humanities, History (author-date variant)
    "Harvard",    # Business, UK Social Sciences
    "Vancouver",  # Medicine, Life Sciences
    "ACM",        # Computer Science
]

# Language filter options: (display_name, iso_code). Empty code = no filter.
LANGUAGES: list[tuple[str, str]] = [
    ("Any", ""),
    ("English", "en"),
    ("German", "de"),
    ("French", "fr"),
    ("Spanish", "es"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Dutch", "nl"),
    ("Russian", "ru"),
    ("Chinese", "zh"),
    ("Japanese", "ja"),
]


@dataclass
class Paper:
    title: str
    authors: list[str]
    year: int | None
    abstract: str | None
    url: str
    source: str                     # "Semantic Scholar", "OpenAlex", …
    doi: str | None = None
    pdf_url: str | None = None
    venue: str | None = None
    citation_count: int | None = None
    tldr: str | None = None         # AI-generated 1-sentence summary (SS only)
    language: str | None = None     # ISO 639-1 code ("en", "de", …) when source provides it
    is_open_access: bool | None = None  # True/False when known, None if unspecified
    is_retracted: bool = False      # set by Crossref when a retraction notice is linked

    def dedup_key(self) -> str:
        if self.doi:
            return f"doi:{self.doi.lower().strip()}"
        return f"title:{self.title.lower().strip()[:100]}"

    def to_bibtex(self, key: str | None = None) -> str:
        """Return a single @article BibTeX entry."""
        return _to_bibtex(self, key)

    def format_citation(self, style: str) -> str:
        t = self.title or "Unknown Title"
        a = self.authors
        y = str(self.year) if self.year else "n.d."
        u = self.url or ""
        v = self.venue or ""

        if style == "APA":
            auth = _apa_authors(a)
            venue_part = f" *{v}*." if v else ""
            return f"{auth} ({y}). {t}.{venue_part} {u}".strip()

        if style == "MLA":
            first = _mla_first(a)
            rest = ", et al." if len(a) > 1 else "."
            venue_part = f" *{v}*," if v else ""
            return f'{first}{rest} "{t}."{venue_part} {y}. {u}'.strip()

        if style == "IEEE":
            auth = _ieee_authors(a)
            venue_part = f", in *{v}*" if v else ""
            return f'{auth}, "{t}"{venue_part}, {y}. [Online]. Available: {u}'.strip()

        if style == "Chicago":
            auth = _chicago_authors(a)
            venue_part = f" *{v}*." if v else ""
            return f'{auth} {y}. "{t}."{venue_part} {u}'.strip()

        if style == "Harvard":
            auth = _harvard_authors(a)
            venue_part = f", *{v}*" if v else ""
            url_part = f". Available at: {u}" if u else ""
            return f"{auth} ({y}) '{t}'{venue_part}{url_part}.".strip()

        if style == "Vancouver":
            auth = _vancouver_authors(a)
            venue_part = f" {v}." if v else ""
            url_part = f" Available from: {u}" if u else ""
            return f"{auth}. {t}.{venue_part} {y}.{url_part}".strip()

        if style == "ACM":
            auth = _acm_authors(a)
            venue_part = f" *{v}*" if v else ""
            return f"{auth}. {y}. {t}.{venue_part} ({y}). {u}".strip()

        return f"{t} ({y})"


# ── BibTeX ────────────────────────────────────────────────────────────────────

def _bibtex_key(p: "Paper") -> str:
    """First-author surname + year + first significant title word."""
    import re as _re
    if p.authors:
        _, surname = _split_name(p.authors[0])
        surname_part = _re.sub(r"\W+", "", surname).lower() or "anon"
    else:
        surname_part = "anon"
    year_part = str(p.year) if p.year else "nd"
    title_word = ""
    if p.title:
        for w in _re.findall(r"\w+", p.title):
            if len(w) > 3 and w.lower() not in ("with", "from", "this", "that", "into"):
                title_word = w.lower()
                break
    parts = [surname_part, year_part]
    if title_word:
        parts.append(title_word)
    return "_".join(parts)


def _bib_escape(value: str) -> str:
    """Escape characters BibTeX treats specially when not inside braces."""
    if not value:
        return ""
    return (
        value.replace("\\", "\\\\")
             .replace("&", r"\&")
             .replace("%", r"\%")
             .replace("$", r"\$")
             .replace("#", r"\#")
             .replace("_", r"\_")
    )


def _bibtex_author(name: str) -> str:
    """BibTeX prefers 'Surname, Given' separated by ' and '."""
    given, surname = _split_name(name)
    if given:
        return f"{surname}, {' '.join(given)}"
    return surname


def _to_bibtex(p: "Paper", key: str | None) -> str:
    cite_key = key or _bibtex_key(p)
    fields: list[str] = []
    if p.title:
        fields.append(f"  title = {{{_bib_escape(p.title)}}}")
    if p.authors:
        fields.append(
            "  author = {"
            + " and ".join(_bib_escape(_bibtex_author(a)) for a in p.authors)
            + "}"
        )
    if p.year:
        fields.append(f"  year = {{{p.year}}}")
    if p.venue:
        fields.append(f"  journal = {{{_bib_escape(p.venue)}}}")
    if p.doi:
        fields.append(f"  doi = {{{p.doi}}}")
    if p.url:
        fields.append(f"  url = {{{p.url}}}")
    return f"@article{{{cite_key},\n" + ",\n".join(fields) + "\n}"


# ── Citation helpers ──────────────────────────────────────────────────────────

def _split_name(n: str) -> tuple[list[str], str]:
    """Return (given_parts, surname). Accepts 'Jane Doe' or 'Doe, Jane'."""
    n = n.strip()
    if "," in n:
        surname, given = [p.strip() for p in n.split(",", 1)]
        return (given.split() if given else []), surname
    parts = n.split()
    if len(parts) < 2:
        return [], n
    return parts[:-1], parts[-1]


def _initials(given: list[str], *, dotted: bool = True, spaced: bool = True) -> str:
    if not given:
        return ""
    letters = [p[0].upper() for p in given if p]
    if dotted:
        return (". " if spaced else ".").join(letters) + "."
    return "".join(letters)


def _apa_authors(names: list[str]) -> str:
    """APA 7: up to 20 authors; for more, list first 19 then '…' then last."""
    if not names:
        return "Unknown"
    fmt = []
    for n in names[:20]:
        given, surname = _split_name(n)
        fmt.append(f"{surname}, {_initials(given)}" if given else surname)
    if len(names) > 20:
        last_given, last_surname = _split_name(names[-1])
        last = f"{last_surname}, {_initials(last_given)}" if last_given else last_surname
        fmt = fmt[:19] + [f"… {last}"]
    if len(fmt) == 1:
        return fmt[0]
    return ", ".join(fmt[:-1]) + f", & {fmt[-1]}"


def _mla_first(names: list[str]) -> str:
    if not names:
        return "Unknown"
    given, surname = _split_name(names[0])
    return f"{surname}, {' '.join(given)}" if given else surname


def _ieee_authors(names: list[str]) -> str:
    result = []
    for n in names[:3]:
        given, surname = _split_name(n)
        if given:
            result.append(f"{_initials(given)} {surname}")
        else:
            result.append(surname)
    return ", ".join(result) + (" et al." if len(names) > 3 else "")


def _chicago_authors(names: list[str]) -> str:
    """Author-date: 'Smith, John, Jane Doe, and Bob Lee.'"""
    if not names:
        return "Unknown."
    formatted = []
    for i, n in enumerate(names[:10]):
        given, surname = _split_name(n)
        full_given = " ".join(given)
        if i == 0:
            formatted.append(f"{surname}, {full_given}".strip(", "))
        else:
            formatted.append(f"{full_given} {surname}".strip())
    if len(names) > 10:
        return ", ".join(formatted) + ", et al."
    if len(formatted) == 1:
        return f"{formatted[0]}."
    return ", ".join(formatted[:-1]) + f", and {formatted[-1]}."


def _harvard_authors(names: list[str]) -> str:
    """'Smith, J., Doe, J. and Lee, B.' (Harvard style, up to 3 authors; else 'et al.')."""
    if not names:
        return "Unknown"
    parts = []
    for n in names[:3]:
        given, surname = _split_name(n)
        parts.append(f"{surname}, {_initials(given, dotted=True, spaced=False)}".rstrip(", "))
    if len(names) > 3:
        return f"{parts[0]} et al."
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _vancouver_authors(names: list[str]) -> str:
    """'Smith J, Doe J, Lee B' — surname + initials without dots, up to 6 authors."""
    parts = []
    for n in names[:6]:
        given, surname = _split_name(n)
        init = _initials(given, dotted=False, spaced=False)
        parts.append(f"{surname} {init}".strip())
    if len(names) > 6:
        parts.append("et al")
    return ", ".join(parts) or "Unknown"


def _acm_authors(names: list[str]) -> str:
    """'John Smith, Jane Doe, and Bob Lee' (full names, Oxford comma)."""
    if not names:
        return "Unknown"
    display = []
    for n in names[:10]:
        given, surname = _split_name(n)
        display.append(f"{' '.join(given)} {surname}".strip())
    if len(names) > 10:
        return ", ".join(display) + ", et al."
    if len(display) == 1:
        return display[0]
    return ", ".join(display[:-1]) + f", and {display[-1]}"
