"""PDF text extraction and resume file loading."""

from pathlib import Path

import fitz  # pymupdf


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF file.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted text content
    """
    doc = fitz.open(pdf_path)
    text_parts = []

    for page in doc:
        text_parts.append(page.get_text())

    doc.close()
    return "\n".join(text_parts)


def extract_text_from_pdf_bytes(data: bytes) -> str:
    """Extract text from PDF bytes."""
    doc = fitz.open(stream=data, filetype="pdf")
    text_parts = []

    for page in doc:
        text_parts.append(page.get_text())

    doc.close()
    return "\n".join(text_parts)


def load_resume_content(path: Path) -> str:
    """Load resume content from a file, extracting text from PDFs."""
    if path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(path)
    return path.read_text()


def load_resume_content_from_upload(filename: str, data: bytes) -> str:
    """Load resume content from uploaded file bytes."""
    if filename.lower().endswith(".pdf"):
        return extract_text_from_pdf_bytes(data)
    return data.decode("utf-8")
