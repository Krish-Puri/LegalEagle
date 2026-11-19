import pytest
from pathlib import Path
from src.agent.reviewer import ContractRiskAgent, _negotiation_note
from src.config import AppConfig
from src.ingest.chunker import chunk_pages
from src.memory.store import MemoryStore
from src.retrieval.vector_store import LocalRetriever
from src.risk.redline import generate_redline
from src.schema import ContractPage, ParsedContract


@pytest.fixture
def test_agent(tmp_path):
    config = AppConfig(
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        db_path=tmp_path / "memory.sqlite3",
        vector_path=tmp_path / "vector",
        playbook_path=Path("data/risk_playbook.yml"),
        labels_path=Path("data/labels.json"),
        classifier_path=tmp_path / "model",
    )
    memory = MemoryStore(config.db_path)
    retriever = LocalRetriever(config.vector_path)
    return ContractRiskAgent(config, retriever, memory)


def test_governing_law_notes_clean():
    # Priority 2: Governing-law findings must NOT contain confidentiality or payment recommendations
    note_crit = _negotiation_note("governing law", "critical", "customer")
    note_high = _negotiation_note("governing law", "high", "customer")
    note_med = _negotiation_note("governing law", "medium", "customer")

    for note in [note_crit, note_high, note_med]:
        lower = note.lower()
        assert "confidential" not in lower, f"Governing law note leaked confidentiality: {note}"
        assert "payment" not in lower and "net 30" not in lower, f"Governing law note leaked payment terms: {note}"
        assert "jurisdiction" in lower or "venue" in lower or "forum" in lower or "law" in lower

    redline = generate_redline("governing law", "This agreement is governed by the laws of Mars.")
    assert "confidential" not in redline.proposed_language.lower()
    assert "jurisdiction" in redline.proposed_language.lower() or "delaware" in redline.proposed_language.lower()


def test_indemnification_notes_clean():
    # Priority 2: Indemnification findings must NOT contain generic termination or liability recommendations
    note_crit = _negotiation_note("indemnification", "critical", "customer")
    note_high = _negotiation_note("indemnification", "high", "customer")

    for note in [note_crit, note_high]:
        lower = note.lower()
        assert "terminat" not in lower, f"Indemnification note leaked termination: {note}"
        assert "indemn" in lower or "third-party" in lower or "defense" in lower

    redline = generate_redline("indemnification", "Customer indemnifies Vendor for all losses.")
    assert "terminat" not in redline.proposed_language.lower()
    assert "indemn" in redline.proposed_language.lower()


def test_perspective_specific_notes():
    # Test customer perspective vs vendor perspective
    customer_note = _negotiation_note("limitation of liability", "critical", "customer")
    vendor_note = _negotiation_note("limitation of liability", "critical", "vendor")

    assert "Priority 1 Redline" in customer_note or "cap" in customer_note
    assert "Vendor Position" in vendor_note or "enterprise" in vendor_note
