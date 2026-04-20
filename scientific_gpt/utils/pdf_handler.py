import PyPDF2
import io


def extract_text_from_pdf(uploaded_file) -> str:
    pdf_bytes = uploaded_file.read()
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text
