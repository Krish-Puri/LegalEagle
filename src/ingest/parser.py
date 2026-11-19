from __future__ import annotations

import re
from pathlib import Path

from src.ingest.chunker import chunk_pages
from src.schema import ContractPage, ParsedContract


SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt"}


def parse_contract(file_path: Path, source_name: str | None = None) -> ParsedContract:
    source_name = source_name or file_path.name
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported contract file type: '{suffix}'. Supported formats are PDF, DOCX, and TXT.")

    if not file_path.exists() or file_path.stat().st_size == 0:
        raise ValueError(f"The selected file '{source_name}' is empty or does not exist.")

    if suffix == ".pdf":
        pages = _parse_pdf(file_path)
    elif suffix == ".docx":
        pages = _parse_docx(file_path)
    else:
        pages = _parse_txt(file_path)

    parsed = ParsedContract(source_name=source_name, pages=pages)
    parsed.chunks = chunk_pages(parsed.pages, parsed.id, parsed.source_name)
    if not parsed.chunks:
        raise ValueError(f"Could not extract reviewable clauses from '{source_name}'. Please verify the contract content.")
    return parsed


def parse_text_contract(text: str, source_name: str = "pasted_contract.txt") -> ParsedContract:
    if not text or not text.strip():
        raise ValueError("Contract text cannot be empty. Please paste or upload a contract.")
    pages = _ensure_pages([ContractPage(page_number=1, text=text, sections=_extract_section_titles(text))])
    parsed = ParsedContract(source_name=source_name, pages=pages)
    parsed.chunks = chunk_pages(parsed.pages, parsed.id, parsed.source_name)
    if not parsed.chunks:
        raise ValueError("No reviewable clauses could be segmented from the provided text.")
    return parsed


def _parse_pdf(file_path: Path) -> list[ContractPage]:
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("Install PyMuPDF (fitz) to parse PDF contracts.") from exc

    try:
        doc = fitz.open(str(file_path))
    except Exception as exc:
        raise ValueError(f"Failed to open PDF document: {exc}") from exc

    pages = []
    for index, page in enumerate(doc, start=1):
        try:
            text = page.get_text("text") or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append(ContractPage(page_number=index, text=text, sections=_extract_section_titles(text)))
    doc.close()
    return _ensure_pages(pages)


def _parse_docx(file_path: Path) -> list[ContractPage]:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("Install python-docx to parse DOCX contracts.") from exc

    try:
        doc = Document(str(file_path))
    except Exception as exc:
        raise ValueError(f"Failed to parse DOCX document: {exc}") from exc

    lines: list[str] = []
    section_titles: list[str] = []
    for paragraph in doc.paragraphs:
        t = paragraph.text.strip()
        if not t:
            continue
        style_name = paragraph.style.name.lower() if paragraph.style else ""
        if "heading" in style_name or (len(t) < 80 and (t.isupper() or any(t.startswith(f"{i}.") for i in range(1, 30)))):
            section_titles.append(t)
        lines.append(paragraph.text)
    text = "\n".join(lines)
    return _ensure_pages([ContractPage(page_number=1, text=text, sections=section_titles or _extract_section_titles(text))])


def _parse_txt(file_path: Path) -> list[ContractPage]:
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        raise ValueError(f"Failed to read TXT file: {exc}") from exc

    page_break = "\f"
    if page_break in text:
        return _ensure_pages(
            [
                ContractPage(page_number=index + 1, text=page, sections=_extract_section_titles(page))
                for index, page in enumerate(text.split(page_break))
                if page.strip()
            ]
        )
    return _ensure_pages([ContractPage(page_number=1, text=text, sections=_extract_section_titles(text))])


def _ensure_pages(pages: list[ContractPage]) -> list[ContractPage]:
    non_empty = [page for page in pages if page.text.strip()]
    if not non_empty:
        raise ValueError("No readable text was found in the contract document.")
    return non_empty


def _extract_section_titles(text: str) -> list[str]:
    titles: list[str] = []
    pattern = re.compile(r"(?m)^\s*(?:\d+(?:\.\d+)*[.)]?\s+)?([A-Z][A-Z\s/&,-]{4,120}|[A-Z][^\n]{4,100})\s*$")
    for match in pattern.finditer(text):
        title = match.group(1).strip()
        if len(title.split()) <= 14 and title not in titles:
            titles.append(title)
    return titles[:20]
