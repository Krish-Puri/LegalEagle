from __future__ import annotations

import re

from src.schema import ContractChunk, ContractPage


HEADING_PATTERN = re.compile(
    r"(?m)(?:^\s*(?P<number>\d+(?:\.\d+)*)(?:[.)])?\s+(?P<title>[A-Z][^\n]{3,120})$|^\s*(?P<allcaps>[A-Z][A-Z\s/&,-]{5,120})$)"
)


def chunk_pages(
    pages: list[ContractPage],
    contract_id: str,
    source_name: str,
    max_chars: int = 1800,
    overlap: int = 180,
) -> list[ContractChunk]:
    chunks: list[ContractChunk] = []
    for page in pages:
        page_text = normalize_text(page.text)
        spans = _clause_spans(page_text)
        if not spans:
            spans = _window_spans(page_text, max_chars=max_chars, overlap=overlap)

        for start, end, clause_number, clause_title in spans:
            text = page_text[start:end].strip()
            if len(text) < 40:
                continue
            if len(text) <= max_chars:
                chunks.append(
                    _make_chunk(
                        contract_id,
                        source_name,
                        page.page_number,
                        start,
                        end,
                        text,
                        page.sections,
                        clause_number,
                        clause_title,
                    )
                )
            else:
                for sub_start, sub_end, _, _ in _window_spans(text, max_chars=max_chars, overlap=overlap):
                    absolute_start = start + sub_start
                    absolute_end = start + sub_end
                    chunks.append(
                        _make_chunk(
                            contract_id,
                            source_name,
                            page.page_number,
                            absolute_start,
                            absolute_end,
                            page_text[absolute_start:absolute_end].strip(),
                            page.sections,
                            clause_number,
                            clause_title,
                        )
                    )
    return chunks


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clause_spans(text: str) -> list[tuple[int, int, str | None, str | None]]:
    heading_spans = _heading_spans(text)
    if heading_spans:
        return heading_spans
    nlp_spans = _spacy_sentence_clause_spans(text)
    if nlp_spans:
        return nlp_spans
    return []


def _heading_spans(text: str) -> list[tuple[int, int, str | None, str | None]]:
    matches = list(HEADING_PATTERN.finditer(text))
    if len(matches) < 2:
        return []
    spans: list[tuple[int, int, str | None, str | None]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        clause_number = match.groupdict().get("number")
        clause_title = match.groupdict().get("title") or match.groupdict().get("allcaps")
        spans.append((start, end, clause_number, clause_title.strip() if clause_title else None))
    prefix = text[: matches[0].start()].strip()
    if prefix:
        spans.insert(0, (0, matches[0].start(), None, "Preamble"))
    return spans


def _spacy_sentence_clause_spans(text: str, max_chars: int = 1800) -> list[tuple[int, int, str | None, str | None]]:
    try:
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
        doc = nlp(text[:100000])
    except Exception:
        return []

    spans: list[tuple[int, int, str | None, str | None]] = []
    current_start: int | None = None
    current_end = 0
    for sent in doc.sents:
        sentence = sent.text.strip()
        if not sentence:
            continue
        if current_start is None:
            current_start = sent.start_char
        current_end = sent.end_char
        boundary = _sentence_looks_like_clause_boundary(sentence)
        if boundary or current_end - current_start >= max_chars:
            spans.append((current_start, current_end, None, _infer_clause_title(text[current_start:current_end])))
            current_start = None
    if current_start is not None and current_end > current_start:
        spans.append((current_start, current_end, None, _infer_clause_title(text[current_start:current_end])))
    return spans if len(spans) > 1 else []


def _sentence_looks_like_clause_boundary(sentence: str) -> bool:
    lower = sentence.lower()
    return any(
        phrase in lower
        for phrase in [
            "provided that",
            "notwithstanding",
            "for the avoidance of doubt",
            "subject to",
            "each party shall",
            "the parties agree",
        ]
    )


def _infer_clause_title(text: str) -> str | None:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    if len(first_line) <= 90 and (first_line.isupper() or re.match(r"^\d+(?:\.\d+)*", first_line)):
        return first_line
    return None


def _window_spans(text: str, max_chars: int, overlap: int) -> list[tuple[int, int, str | None, str | None]]:
    if len(text) <= max_chars:
        return [(0, len(text), None, _infer_clause_title(text))]
    spans: list[tuple[int, int, str | None, str | None]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        spans.append((start, end, None, _infer_clause_title(text[start:end])))
        if end == len(text):
            break
        start = max(0, end - overlap)
    return spans


def _make_chunk(
    contract_id: str,
    source_name: str,
    page: int,
    start: int,
    end: int,
    text: str,
    section_path: list[str],
    clause_number: str | None,
    clause_title: str | None,
) -> ContractChunk:
    return ContractChunk(
        contract_id=contract_id,
        source_name=source_name,
        page=page,
        section_path=section_path,
        clause_number=clause_number,
        clause_title=clause_title,
        start_offset=start,
        end_offset=end,
        text=text,
    )
