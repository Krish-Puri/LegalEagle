from src.ingest.chunker import chunk_pages
from src.schema import ContractPage


def test_chunk_pages_preserves_offsets():
    text = "1. TERMINATION\nVendor may terminate immediately without notice.\n\n2. CONFIDENTIALITY\nEach party shall protect confidential information."
    chunks = chunk_pages([ContractPage(page_number=1, text=text)], "c1", "sample.txt")

    assert chunks
    first = chunks[0]
    assert first.page == 1
    assert text[first.start_offset : first.end_offset].strip().startswith("1. TERMINATION")
