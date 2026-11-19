from src.agent.reviewer import ContractRiskAgent
from src.config import AppConfig
from src.ingest.chunker import chunk_pages
from src.memory.store import MemoryStore
from src.retrieval.vector_store import LocalRetriever
from src.schema import ContractPage, ParsedContract


def test_offline_agent_finds_risks(tmp_path):
    config = AppConfig(
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        db_path=tmp_path / "memory.sqlite3",
        vector_path=tmp_path / "vector",
        playbook_path=tmp_path / "missing.yml",
        labels_path=tmp_path / "labels.json",
        classifier_path=tmp_path / "model",
    )
    memory = MemoryStore(config.db_path)
    retriever = LocalRetriever(config.vector_path)
    parsed = ParsedContract(
        source_name="sample.txt",
        pages=[ContractPage(page_number=1, text="Vendor may terminate immediately without notice.")],
    )
    parsed.chunks = chunk_pages(parsed.pages, parsed.id, parsed.source_name)

    scorecard = ContractRiskAgent(config, retriever, memory).analyze(parsed)

    assert scorecard.findings
    assert scorecard.overall_risk_score > 0
