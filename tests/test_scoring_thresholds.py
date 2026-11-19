import pytest
from src.risk.scoring import evaluate_clause_risk, overall_score, score_to_level
from src.schema import RiskFinding


def test_score_to_level_thresholds():
    # Canonical thresholds: Low: 0-39, Medium: 40-69, High: 70-87, Critical: 88-100
    assert score_to_level(0) == "low"
    assert score_to_level(39) == "low"
    assert score_to_level(40) == "medium"
    assert score_to_level(69) == "medium"
    assert score_to_level(70) == "high"
    assert score_to_level(87) == "high"
    assert score_to_level(88) == "critical"
    assert score_to_level(100) == "critical"


def test_critical_risk_floor_override():
    # If any finding is Critical, overall_score MUST be at least 88
    critical_finding = RiskFinding(
        clause_type="limitation of liability",
        quote="Customer liability is unlimited. Vendor liability capped at $50.",
        page=1,
        start_offset=0,
        end_offset=70,
        business_impact="Catastrophic unhedged financial liability.",
        risk_level="critical",
        category="liability_exposure",
        risk_score=98,
        review_priority="critical",
        rationale="Uncapped exposure",
        suggested_negotiation_note="Cap liability",
    )
    benign_finding_1 = RiskFinding(
        clause_type="confidentiality",
        quote="Mutual confidentiality.",
        page=1,
        start_offset=71,
        end_offset=95,
        business_impact="Standard commercial confidentiality.",
        risk_level="low",
        category="information_protection",
        risk_score=15,
        review_priority="low",
        rationale="Standard",
        suggested_negotiation_note="Acceptable",
    )
    benign_finding_2 = RiskFinding(
        clause_type="governing law",
        quote="Governed by Delaware law.",
        page=1,
        start_offset=96,
        end_offset=125,
        business_impact="Predictable commercial jurisdiction.",
        risk_level="low",
        category="dispute_resolution",
        risk_score=20,
        review_priority="low",
        rationale="Standard",
        suggested_negotiation_note="Acceptable",
    )

    # Even with many low findings, the single critical finding MUST guarantee a floor >= 88
    findings = [critical_finding, benign_finding_1, benign_finding_2]
    agg_score = overall_score(findings, missing=[])

    assert agg_score >= 88, f"Expected aggregate score >= 88, got {agg_score}"
    assert score_to_level(agg_score) == "critical"


def test_perspective_divergence_scoring():
    clause_text = (
        "Customer's liability for payment obligations and breaches is unlimited. "
        "Vendor's total cumulative aggregate liability shall not exceed fifty dollars ($50.00)."
    )

    # Customer perspective
    cust_score, cust_level, cust_triggers, _ = evaluate_clause_risk(
        clause_type="limitation of liability",
        clause_text=clause_text,
        base_level="high",
        confidence=0.95,
        reviewing_party="customer",
    )
    assert cust_score >= 88, f"Customer score expected >= 88, got {cust_score}"
    assert cust_level == "critical"

    # Vendor perspective: protective terms for vendor means minimal operational exposure
    vend_score, vend_level, vend_triggers, _ = evaluate_clause_risk(
        clause_type="limitation of liability",
        clause_text=clause_text,
        base_level="high",
        confidence=0.95,
        reviewing_party="vendor",
    )
    assert vend_score <= 39, f"Vendor score expected <= 39, got {vend_score}"
    assert vend_level == "low"
