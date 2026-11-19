from src.memory.store import MemoryStore
from src.schema import ContractPage, ParsedContract, RiskScorecard


def test_memory_roundtrip(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    parsed = ParsedContract(source_name="x.txt", pages=[ContractPage(page_number=1, text="hello")])
    scorecard = RiskScorecard(contract_summary="ok", overall_risk_score=5, risk_level="low")

    contract_id = store.save_contract_analysis(parsed, scorecard)
    store.save_feedback(contract_id, "accepted", "Looks right.")
    store.save_preferences("Prefer strict liability review.")

    assert store.list_contracts()[0]["id"] == parsed.id
    assert "Looks right" in store.feedback_context()
    assert "strict liability" in store.get_preferences()
