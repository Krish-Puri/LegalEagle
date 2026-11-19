from __future__ import annotations

from pathlib import Path

from src.agent.reviewer import ContractRiskAgent
from src.config import AppConfig
from src.evals.runner import run_comprehensive_evaluation
from src.memory.store import MemoryStore
from src.retrieval.vector_store import LocalRetriever


def test_comprehensive_eval_runner(tmp_path):
    config = AppConfig.from_env()
    memory = MemoryStore(tmp_path / "memory.sqlite3")
    retriever = LocalRetriever(tmp_path / "vector", config=config)
    agent = ContractRiskAgent(config, retriever, memory)

    results = run_comprehensive_evaluation(agent, Path("data/eval_dataset.json"))

    assert "classification_metrics" in results
    assert "retrieval_metrics" in results
    assert "structured_output_metrics" in results
    assert "end_to_end_review" in results

    # Macro F1 should be high on the benchmark dataset
    macro_f1 = results["classification_metrics"]["macro_f1"]
    assert macro_f1 >= 0.85, f"Expected Macro-F1 >= 0.85, got {macro_f1}"

    # Retrieval Hit@3 should be strong
    hit3 = results["retrieval_metrics"]["hit@3"]
    assert hit3 >= 0.75, f"Expected Hit@3 >= 0.75, got {hit3}"

    # Finding recall should be high
    recall = results["end_to_end_review"]["finding_recall"]
    assert recall >= 0.80, f"Expected finding recall >= 0.80, got {recall}"

    # Verify saved report
    report_file = Path("research_evidence/evaluation_results/benchmark_report.json")
    assert report_file.exists()

