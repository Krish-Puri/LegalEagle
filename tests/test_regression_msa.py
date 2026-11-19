from __future__ import annotations

from pathlib import Path

from src.agent.reviewer import ContractRiskAgent
from src.config import AppConfig
from src.ingest.parser import parse_contract
from src.memory.store import MemoryStore
from src.retrieval.vector_store import LocalRetriever


def test_master_services_agreement_customer_perspective(tmp_path):
    contract_path = Path("data/demo_evaluation_contract.txt")
    assert contract_path.exists(), "demo_evaluation_contract.txt fixture missing"

    parsed = parse_contract(contract_path)
    assert len(parsed.chunks) >= 8, f"Expected at least 8 clauses, found {len(parsed.chunks)}"

    config = AppConfig.from_env()
    memory = MemoryStore(tmp_path / "memory.sqlite3")
    retriever = LocalRetriever(tmp_path / "vector", config=config)
    agent = ContractRiskAgent(config, retriever, memory)

    # 1. Analyze from Customer perspective
    scorecard_customer = agent.analyze(parsed, reviewing_party="customer")

    findings = {f.clause_type: f for f in scorecard_customer.findings}

    # Regression Assertion 1: IP Assignment vs Contract Assignment separation
    assert "ip assignment" in findings, "Clause 4 must be classified as 'ip assignment'"
    assert "assignment" in findings, "Clause 7 must be classified as 'assignment'"

    ip_finding = findings["ip assignment"]
    assign_finding = findings["assignment"]

    # Verify IP finding does NOT have assignment text
    assert "deliverables" in ip_finding.quote.lower() or "irrevocably assigns" in ip_finding.quote.lower()
    assert ip_finding.category == "ip_ownership"
    assert "deliverable" in ip_finding.redline.proposed_language.lower() or "custom" in ip_finding.redline.proposed_language.lower()
    assert "affiliate or successor in connection with a merger" not in ip_finding.redline.proposed_language.lower()

    # Verify Agreement assignment finding has transfer text
    assert "assign or transfer this agreement" in assign_finding.quote.lower()
    assert assign_finding.category == "operational_control"
    assert "merger" in assign_finding.redline.proposed_language.lower() or "affiliate" in assign_finding.redline.proposed_language.lower()

    # Regression Assertion 2: Liability recognized as Critical risk with asymmetric $50 cap
    assert "limitation of liability" in findings, "Clause 2 must be classified as 'limitation of liability'"
    liability_finding = findings["limitation of liability"]
    assert liability_finding.risk_level == "critical", f"Expected critical liability risk, got {liability_finding.risk_level}"
    assert liability_finding.risk_score >= 90, f"Expected liability risk score >= 90, got {liability_finding.risk_score}"
    assert any("unlimited" in f.lower() or "$50" in f.lower() or "nominal" in f.lower() for f in liability_finding.risk_factors)

    # Regression Assertion 3: Unilateral termination recognized
    assert "termination" in findings, "Clause 1 must be classified as 'termination'"
    term_finding = findings["termination"]
    assert term_finding.risk_level == "critical", f"Expected critical termination risk, got {term_finding.risk_level}"
    assert any("immediate" in f.lower() or "notice" in f.lower() for f in term_finding.risk_factors)

    # Regression Assertion 4: Broad indemnity recognized
    assert "indemnification" in findings, "Clause 3 must be classified as 'indemnification'"
    indem_finding = findings["indemnification"]
    assert indem_finding.risk_level in {"high", "critical"}

    # Regression Assertion 5: 5-year non-compete recognized
    assert "non-compete / non-solicit" in findings, "Clause 5 must be classified as 'non-compete / non-solicit'"
    nc_finding = findings["non-compete / non-solicit"]
    assert nc_finding.risk_level in {"high", "critical"}

    # Regression Assertion 6: Confidentiality categorized under information_protection (not ip_risk)
    assert "confidentiality" in findings, "Clause 6 must be classified as 'confidentiality'"
    conf_finding = findings["confidentiality"]
    assert conf_finding.category == "information_protection", f"Expected information_protection, got {conf_finding.category}"

    # Regression Assertion 7: Overall risk score must be Critical or High (never diluted away by benign clauses)
    assert scorecard_customer.risk_level in {"critical", "high"}
    assert scorecard_customer.overall_risk_score >= 75

    # 2. Analyze from Vendor perspective and check divergence
    scorecard_vendor = agent.analyze(parsed, reviewing_party="vendor")
    vendor_findings = {f.clause_type: f for f in scorecard_vendor.findings}

    # Vendor perspective: liability clause protects Vendor ($50 cap), so Vendor's operational risk is Low/Medium
    if "limitation of liability" in vendor_findings:
        v_liab = vendor_findings["limitation of liability"]
        assert v_liab.risk_score < liability_finding.risk_score, "Vendor liability risk score must be lower than Customer's"
        assert v_liab.risk_level in {"low", "medium"}

    if "termination" in vendor_findings:
        v_term = vendor_findings["termination"]
        assert v_term.risk_score < term_finding.risk_score, "Vendor termination risk score must be lower than Customer's"

