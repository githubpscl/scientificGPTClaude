import io


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from a PDF.

    Tries pdfplumber first (better column / table handling, fewer broken words);
    falls back to PyPDF2 if pdfplumber is unavailable or fails on a given file.
    """
    pdf_bytes = uploaded_file.read()
    text = _extract_with_pdfplumber(pdf_bytes)
    if text and text.strip():
        return text
    return _extract_with_pypdf2(pdf_bytes)


def _extract_with_pdfplumber(pdf_bytes: bytes) -> str:
    try:
        import pdfplumber
    except ImportError:
        return ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)
    except Exception:
        return ""


def _extract_with_pypdf2(pdf_bytes: bytes) -> str:
    import PyPDF2
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    return "".join((page.extract_text() or "") for page in reader.pages)
