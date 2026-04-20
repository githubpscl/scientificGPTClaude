from __future__ import annotations
from dataclasses import dataclass, field


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

    def dedup_key(self) -> str:
        if self.doi:
            return f"doi:{self.doi.lower().strip()}"
        return f"title:{self.title.lower().strip()[:100]}"

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
            rest = ", et al." if len(a) > 1 else ""
            venue_part = f" *{v}*," if v else ""
            return f'{first}{rest}. "{t}."{venue_part} {y}. {u}'.strip()

        if style == "IEEE":
            auth = _ieee_authors(a)
            venue_part = f", in *{v}*" if v else ""
            return f'{auth}, "{t}"{venue_part}, {y}. [Online]. Available: {u}'.strip()

        return f"{t} ({y})"


# ── Citation helpers ──────────────────────────────────────────────────────────

def _apa_authors(names: list[str]) -> str:
    fmt = []
    for n in names[:7]:
        parts = n.split()
        if len(parts) >= 2:
            ini = ". ".join(p[0].upper() for p in parts[:-1]) + "."
            fmt.append(f"{parts[-1]}, {ini}")
        else:
            fmt.append(n)
    if len(names) > 7:
        fmt = fmt[:6] + [f"… {fmt[-1]}"]
    return ", ".join(fmt) or "Unknown"


def _mla_first(names: list[str]) -> str:
    if not names:
        return "Unknown"
    parts = names[0].split()
    return f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) >= 2 else names[0]


def _ieee_authors(names: list[str]) -> str:
    result = []
    for n in names[:3]:
        parts = n.split()
        if len(parts) >= 2:
            ini = ". ".join(p[0].upper() for p in parts[:-1])
            result.append(f"{ini}. {parts[-1]}")
        else:
            result.append(n)
    return ", ".join(result) + (" et al." if len(names) > 3 else "")
