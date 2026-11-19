import pytest
from src.agent.reviewer import ContractRiskAgent
from src.agent.conversation import LegalEagleChatAgent
from src.config import AppConfig
from src.memory.store import MemoryStore
from src.retrieval.vector_store import LocalRetriever
from src.schema import ContractChunk, ContractPage, ParsedContract


@pytest.fixture
def test_setup(tmp_path):
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
    agent = ContractRiskAgent(config, retriever, memory)
    chat_agent = LegalEagleChatAgent(agent, memory)
    return agent, chat_agent, memory, retriever


from pathlib import Path


def test_termination_precedent_integrity(test_setup):
    agent, chat_agent, _, retriever = test_setup

    # Precedent search for termination should only return termination precedents
    precedents = retriever.search(
        "Vendor may terminate immediately without notice",
        k=5,
        source_types=["playbook", "market_precedent"],
        clause_type="termination",
    )

    assert precedents, "Expected to retrieve termination playbook/precedent entries"
    for prec in precedents:
        cat = (prec.clause_category or prec.metadata.get("clause_type") or "").lower()
        title = prec.document_title.lower()
        assert "termination" in cat or "termination" in title, (
            f"Cross-category leakage! Found {cat} / {title} in termination search"
        )
        assert "confidential" not in cat
        assert "intellectual property" not in cat


def test_liability_finding_no_ip_precedent(test_setup):
    agent, _, _, _ = test_setup

    raw = (
        "Customer liability for payment obligations is unlimited. "
        "Vendor aggregate cumulative liability shall not exceed fifty dollars ($50.00)."
    )
    parsed = ParsedContract(
        source_name="liability_test.txt",
        pages=[ContractPage(page_number=1, text=raw)],
    )
    from src.ingest.chunker import chunk_pages
    parsed.chunks = chunk_pages(parsed.pages, parsed.id, parsed.source_name)

    scorecard = agent.analyze(parsed, reviewing_party="customer")
    liability_findings = [f for f in scorecard.findings if f.clause_type == "limitation of liability"]

    assert liability_findings, "Expected limitation of liability finding"
    for f in liability_findings:
        if f.market_standard:
            doc_title = f.market_standard.source_title.lower()
            summary = f.market_standard.summary.lower()
            assert "intellectual property" not in doc_title and "ip assignment" not in doc_title, (
                f"Liability finding retrieved IP precedent: {f.market_standard.source_title}"
            )
            assert "patent" not in summary and "deliverable" not in summary


def test_chat_termination_no_confidentiality_citation(test_setup):
    agent, chat_agent, _, _ = test_setup

    raw = "Vendor may terminate this Agreement immediately without notice if invoice is unpaid."
    parsed = ParsedContract(
        source_name="termination_contract.txt",
        pages=[ContractPage(page_number=1, text=raw)],
    )
    from src.ingest.chunker import chunk_pages
    parsed.chunks = chunk_pages(parsed.pages, parsed.id, parsed.source_name)

    ans = chat_agent.answer(
        question="Can the Vendor terminate without notice?",
        parsed=parsed,
        session_id="test_qa_session",
    )

    assert ans.grounded
    # Ensure no confidentiality citation appears
    for cite in ans.citations:
        assert "confidentiality" not in cite.lower(), f"Unrelated confidentiality citation in termination answer: {cite}"
    # Answer text should not discuss confidentiality
    assert "confidential" not in ans.answer.lower()
